# agent_trainer.py
from typing import Dict, Any
import torch
import numpy as np
import uuid
from copy import deepcopy

# Import from verl
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.utils.debug.performance import marked_timer

# Import our multi-turn rollout manager
from rollout.sync_multi_turn_rollout import SyncMultiTurnRollout


class AgentTrainer(RayPPOTrainer):
    """
    Agent Trainer that inherits from RayPPOTrainer and replaces single-turn generation 
    with multi-turn rollout logic using SyncMultiTurnRollout.
    
    Key modifications:
    1. Replaces self.actor_rollout_wg.generate_sequences() with SyncMultiTurnRollout
    2. Maintains compatibility with existing PPO training pipeline
    3. Supports agent-based multi-turn interactions via registry system
    4. Agent class is automatically resolved from config
    """

    def __init__(self, config, tokenizer, role_worker_mapping, resource_pool_manager, 
                 processor=None, **kwargs):
        """
        Initialize AgentTrainer. Agent class is automatically determined from config.
        
        Args:
            config: Configuration containing agent specification
            tokenizer: Tokenizer for text processing
            role_worker_mapping: Role to worker mapping
            resource_pool_manager: Resource pool manager
            processor: Optional data processor
            **kwargs: Additional arguments passed to parent RayPPOTrainer
        """
        # Initialize parent PPO trainer
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            processor=processor,
            **kwargs
        )
        
        # Initialize multi-turn rollout manager (will be created after init_workers)
        self.multi_turn_rollout = None

    def init_workers(self):
        """Override to initialize both parent workers and multi-turn rollout manager."""
        # Initialize parent workers first
        super().init_workers()
        
    def init_multi_turn_rollout(self):
        # Initialize multi-turn rollout manager - agent_cls will be resolved from config
        self.multi_turn_rollout = SyncMultiTurnRollout(
            actor_rollout_wg=self.actor_rollout_wg,
            cfg=self.config,
            tokenizer=self.tokenizer,
            processor=self.processor
        )

    def _generate_multi_turn_sequences(self, gen_batch: DataProto) -> DataProto:
        """
        Replace single-turn generation with multi-turn rollout logic.
        
        Args:
            gen_batch: DataProto containing prompts for generation
            
        Returns:
            DataProto: Generated sequences in format compatible with PPO pipeline
        """
        # Set meta info for generation
        gen_batch.meta_info.update({
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": True,
        })
        
        # Reset rollout manager for new batch
        if hasattr(self.multi_turn_rollout, 'reset'):
            self.multi_turn_rollout.reset()
        
        # Perform multi-turn rollout
        final_env_outs = self.multi_turn_rollout.rollout()
        
        # Build update batch for PPO training
        rollout_batch = self.multi_turn_rollout.build_update_batch()
        
        # Convert rollout_batch to format expected by PPO trainer
        output_batch = DataProto(
            batch={
                "responses": rollout_batch.batch.get("responses", gen_batch.batch["input_ids"]),
                "sequences": rollout_batch.batch.get("sequences", gen_batch.batch["input_ids"]),
                "attention_mask": rollout_batch.batch.get("attention_mask", gen_batch.batch.get("attention_mask")),
                "position_ids": rollout_batch.batch.get("position_ids", gen_batch.batch.get("position_ids")),
            },
            non_tensor_batch=rollout_batch.non_tensor_batch,
            meta_info=rollout_batch.meta_info
        )
        
        return output_batch

    def fit(self):
        """
        Override fit method to use multi-turn generation instead of single-turn.
        Only modifies the generation step - everything else remains the same.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # Load checkpoint before doing anything
        self._load_checkpoint()

        # Perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            print(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # Add progress bar
        from tqdm import tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # Start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                # Check if profiling is enabled
                do_profile = (self.config.trainer.profile_steps is not None and 
                             self.global_steps in self.config.trainer.profile_steps)
                
                if do_profile:
                    self.actor_rollout_wg.start_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.start_profile()
                    if self.use_critic:
                        self.critic_wg.start_profile()
                    if self.use_rm:
                        self.rm_wg.start_profile()

                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # Pop keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # ═══════════════════════════════════════════════════════════
                    # MAIN MODIFICATION: Replace single-turn with multi-turn
                    # ═══════════════════════════════════════════════════════════
                    with marked_timer("gen", timing_raw, color="red"):
                        # Use multi-turn rollout instead of single-turn generation
                        gen_batch_output = self._generate_multi_turn_sequences(gen_batch)
                        
                        # Update timing if available
                        if "timing" in gen_batch_output.meta_info:
                            timing_raw.update(gen_batch_output.meta_info["timing"])
                            gen_batch_output.meta_info.pop("timing", None)

                    # Handle REMAX advantage estimator if needed
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            
                            # For baseline, fall back to original method
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # Add unique IDs and continue with standard PPO pipeline
                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    
                    # Repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # Continue with standard PPO pipeline from parent class
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    
                    # Balance batch if needed
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # Compute global valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # Compute rewards
                    if self.use_rm:
                        with marked_timer("reward", timing_raw, color="green"):
                            batch = self.rm_wg.compute_reward(batch)
                    elif self.reward_fn is not None:
                        with marked_timer("reward", timing_raw, color="green"):
                            reward_tensor = self.reward_fn(batch)
                            batch.batch["rewards"] = reward_tensor

                    # Update critic if needed
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        from verl.trainer.ppo.metric_utils import reduce_metrics
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # Update actor (after critic warmup)
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = True  # Enable multi-turn for actor update
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        from verl.trainer.ppo.metric_utils import reduce_metrics
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Validation and checkpointing (same as parent)
                    if (self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and 
                        (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if (self.config.trainer.save_freq > 0 and 
                        (is_last_step or self.global_steps % self.config.trainer.save_freq == 0)):
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                # Add training metrics
                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })

                # Log metrics
                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1

                # Stop profiling if enabled
                if do_profile:
                    self.actor_rollout_wg.stop_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.stop_profile()
                    if self.use_critic:
                        self.critic_wg.stop_profile()
                    if self.use_rm:
                        self.rm_wg.stop_profile()

                # Break if last step
                if is_last_step:
                    print(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
