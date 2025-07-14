# agent_trainer.py
from typing import Dict, Any
import torch
import numpy as np
import uuid
import ray
from copy import deepcopy
from pprint import pprint
from tqdm import tqdm

# Import from verl
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, apply_kl_penalty, compute_advantage
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_timing_metrics, compute_throughout_metrics
from verl.utils.debug.performance import marked_timer
from verl.utils.metric import reduce_metrics

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

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Override to avoid creating traditional dataloaders since we generate data dynamically
        through multi-turn rollouts.
        """
        # Ensure total training steps is configured
        assert self.config.trainer.total_training_steps is not None, "must determine total training steps"
        
        self.total_training_steps = self.config.trainer.total_training_steps
        print(f"Total training steps: {self.total_training_steps}")
        
        # Create a dummy dataloader that yields empty batches
        # The actual data generation happens in _generate_multi_turn_sequences
        from torch.utils.data import Dataset
        
        class DummyDataset(Dataset):
            def __init__(self, total_steps):
                self.total_steps = total_steps
                
            def __len__(self):
                return self.total_steps
            
            def __getitem__(self, idx):
                return {}  # Empty dict since we don't use this data
        
        from torch.utils.data import DataLoader
        
        # Create minimal dataloader just to satisfy the training loop structure
        self.train_dataloader = DataLoader(
            dataset=DummyDataset(self.total_training_steps),
            batch_size=1,  # Doesn't matter since we ignore the data
            shuffle=False
        )
        
        # Set validation dataloader to None or create a dummy one for validation
        self.val_dataloader = None
        
        # Update optimizer configs with total training steps
        try:
            from omegaconf import OmegaConf, open_dict
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = self.total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = self.total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def init_workers(self):
        """Override to initialize both parent workers and multi-turn rollout manager."""
        # Initialize parent workers first
        super().init_workers()
        
    def init_multi_turn_rollout(self):
        """Initialize multi-turn rollout manager."""
        # Initialize multi-turn rollout manager - agent_cls will be resolved from config
        self.multi_turn_rollout = SyncMultiTurnRollout(
            actor_rollout_wg=self.actor_rollout_wg,
            cfg=self.config,
            tokenizer=self.tokenizer,
            processor=self.processor
        )

    def _filter_rollout(self, batch: DataProto):
        """
        Filter rollout based on in-group statistics. We want groups with high-quality rollouts
        that deviate significantly from the mean.
        
        Args:
            batch: DataProto containing rollout data
            
        Returns:
            tuple: (filtered_batch, metrics_dict)
        """
        rollout_filter_ratio = getattr(self.config.rollout, 'rollout_filter_ratio', 1.0)
        rollout_filter_type = getattr(self.config.rollout, 'rollout_filter_type', 'std')
        
        # Use rollout config instead of es_manager config
        num_groups = self.config.rollout.agent_group_num
        group_size = self.config.rollout.agent_group_size

        # Get RM scores and reshape to groups
        rm_scores = batch.batch["rm_scores"].sum(dim=-1).view(num_groups, group_size)
        
        # Calculate in-group statistics
        in_group_std = rm_scores.std(dim=-1)
        in_group_max = rm_scores.max(dim=-1).values
        in_group_mean = rm_scores.mean(dim=-1)
        
        # If no filtering (ratio = 1), return all data
        if rollout_filter_ratio >= 1.0:
            metrics = {
                "rollout/in_group_std": in_group_std.mean(),
                "rollout/in_group_max": in_group_max.mean(),
                "rollout/in_group_mean": in_group_mean.mean(),
                "rollout/chosen_in_group_std": in_group_std.mean(),
                "rollout/chosen_in_group_max": in_group_max.mean(),
                "rollout/chosen_in_group_mean": in_group_mean.mean()
            }
            return batch, metrics

        # Select top groups based on filter type
        num_groups_to_keep = max(1, int(rollout_filter_ratio * num_groups))
        
        if rollout_filter_type == "std_rev":
            top_groups = (-in_group_std).topk(num_groups_to_keep).indices
        elif rollout_filter_type == "std":
            top_groups = in_group_std.topk(num_groups_to_keep).indices
        else:
            raise ValueError(f"Invalid rollout filter type: {rollout_filter_type}")

        # Create mask for selected groups
        mask = torch.zeros(num_groups, dtype=torch.bool)
        mask[top_groups] = True
        mask = mask.unsqueeze(1).expand(-1, group_size).flatten()

        # Filter batch tensors - maintain TensorDict structure
        filtered_batch = DataProto()
        filtered_tensor_dict = {}
        
        for key, value in batch.batch.items():
            if isinstance(value, torch.Tensor):
                filtered_tensor_dict[key] = value[mask]
            else:
                filtered_tensor_dict[key] = value
        
        # Create new TensorDict with filtered tensors
        try:
            from tensordict import TensorDict
            filtered_batch.batch = TensorDict(
                filtered_tensor_dict,
                batch_size=mask.sum().item(),
                device=next(iter(filtered_tensor_dict.values())).device if filtered_tensor_dict else None
            )
        except ImportError:
            # Fallback to regular dict if TensorDict not available
            filtered_batch.batch = filtered_tensor_dict

        # Filter non-tensor batch
        filtered_batch.non_tensor_batch = {}
        for key, value in batch.non_tensor_batch.items():
            if isinstance(value, np.ndarray):
                filtered_batch.non_tensor_batch[key] = value[mask.cpu().numpy()]
            elif isinstance(value, list):
                filtered_batch.non_tensor_batch[key] = [v for v, m in zip(value, mask.cpu().numpy()) if m]
            else:
                filtered_batch.non_tensor_batch[key] = value

        # Copy meta_info
        filtered_batch.meta_info = batch.meta_info.copy() if batch.meta_info else {}

        # Calculate metrics
        metrics = {
            "rollout/in_group_std": in_group_std.mean(),
            "rollout/in_group_max": in_group_max.mean(),
            "rollout/in_group_mean": in_group_mean.mean(),
            "rollout/chosen_in_group_std": in_group_std[top_groups].mean(),
            "rollout/chosen_in_group_max": in_group_max[top_groups].mean(),
            "rollout/chosen_in_group_mean": in_group_mean[top_groups].mean(),
            "rollout/filter_ratio_actual": len(top_groups) / num_groups,
            "rollout/groups_kept": len(top_groups),
            "rollout/total_groups": num_groups
        }

        return filtered_batch, metrics

    def _generate_multi_turn_sequences(self, gen_batch: DataProto) -> tuple[DataProto, dict]:
        """
        Replace single-turn generation with multi-turn rollout logic.
        
        Args:
            gen_batch: DataProto containing prompts for generation (mostly metadata)
            
        Returns:
            tuple: (rollout_batch: DataProto, filter_metrics: dict)
        """
        print(f"\n🔍 DEBUG: _generate_multi_turn_sequences called")
        print(f"   Gen batch type: {type(gen_batch)}")
        print(f"   Gen batch has batch: {hasattr(gen_batch, 'batch')}")
        print(f"   Gen batch has non_tensor_batch: {hasattr(gen_batch, 'non_tensor_batch')}")
        
        # Initialize multi-turn rollout if not already done
        if self.multi_turn_rollout is None:
            print(f"   Initializing multi-turn rollout...")
            self.init_multi_turn_rollout()
            print(f"   Multi-turn rollout initialized")
        
        # Type narrowing assertion - we know it's not None after initialization
        assert self.multi_turn_rollout is not None, "multi_turn_rollout should be initialized"
        
        print(f"   Multi-turn rollout info:")
        print(f"      Number of agents: {self.multi_turn_rollout.n_agents}")
        print(f"      Agent group num: {self.multi_turn_rollout.agent_group_num}")  
        print(f"      Agent group size: {self.multi_turn_rollout.agent_group_size}")
        print(f"      Max turns: {self.multi_turn_rollout.max_turns}")
        print(f"      Current step: {self.multi_turn_rollout.step_cnt}")
        
        # Run multi-turn rollout to get complete trajectories
        print(f"   Starting multi-turn rollout...")
        final_env_outs = self.multi_turn_rollout.rollout()
        print(f"   Multi-turn rollout completed")
        print(f"   Final env outs type: {type(final_env_outs)}")
        print(f"   Final env outs length: {len(final_env_outs) if final_env_outs else 0}")
        
        if final_env_outs:
            done_count = sum(1 for env_out in final_env_outs if env_out.done)
            print(f"   Done agents: {done_count}/{len(final_env_outs)}")
        
        # Build update batch containing full trajectories and rewards
        # This already returns a complete DataProto with all necessary fields
        print(f"   Building PPO batch...")
        rollout_batch = self.multi_turn_rollout.build_ppo_batch()
        print(f"   PPO batch built")
        print(f"   PPO batch type: {type(rollout_batch)}")
        print(f"   PPO batch has batch: {hasattr(rollout_batch, 'batch')}")
        
        if hasattr(rollout_batch, 'batch') and rollout_batch.batch is not None:
            print(f"   PPO batch keys: {list(rollout_batch.batch.keys())}")
            if 'input_ids' in rollout_batch.batch:
                input_ids = rollout_batch.batch['input_ids']
                print(f"   Input IDs shape: {input_ids.shape}")
                
                # Check for all-padding sequences in the final batch
                pad_token_id = self.tokenizer.pad_token_id
                problematic_agents = []
                for i in range(input_ids.shape[0]):
                    seq = input_ids[i]
                    non_pad_count = (seq != pad_token_id).sum().item()
                    if non_pad_count == 0:
                        problematic_agents.append(i)
                
                if problematic_agents:
                    print(f"   ❌ CRITICAL: Final PPO batch has {len(problematic_agents)} all-padding sequences!")
                    print(f"   This will cause vLLM IndexError!")
        
        # Apply rollout filtering if enabled
        rollout_filter_ratio = getattr(self.config.rollout, 'rollout_filter_ratio', 1.0)
        if rollout_filter_ratio < 1.0:
            print(f"   Applying rollout filtering (ratio: {rollout_filter_ratio})...")
            rollout_batch, filter_metrics = self._filter_rollout(rollout_batch)
            # Add filter metrics to batch meta_info
            if rollout_batch.meta_info is None:
                rollout_batch.meta_info = {}
            rollout_batch.meta_info.update(filter_metrics)
            print(f"   Rollout filtering applied")
        else:
            print(f"   No rollout filtering (ratio: {rollout_filter_ratio})")
            filter_metrics = {}
        
        print(f"   Returning rollout batch and metrics")
        # Return the complete DataProto and metrics as tuple consistently
        return rollout_batch, filter_metrics

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
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

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                do_profile = self.global_steps in self.config.trainer.profile_steps if self.config.trainer.profile_steps is not None else False
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
                
                # ─────────────────── MAIN MODIFICATION: IGNORE BATCH_DICT ───────────────────
                # Instead of: batch = DataProto.from_single_dict(batch_dict)
                # We create an empty batch since data comes from rollouts
                batch: DataProto = DataProto()

                # Skip the pop() operations since we don't have input data
                # gen_batch will be empty, actual generation happens in _generate_multi_turn_sequences
                gen_batch = DataProto()

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # ─────────────────── MULTI-TURN ROLLOUT GENERATION ───────────────────
                    # generate a batch using multi-turn rollout
                    with marked_timer("gen", timing_raw, color="red"):
                        # Use multi-turn rollout instead of single-turn generation
                        batch, rollout_metrics = self._generate_multi_turn_sequences(gen_batch)
                        
                        # Add rollout metrics to training metrics
                        metrics.update(rollout_metrics)
                        metrics.update({"train/" + key: value for key, value in batch.meta_info['metrics'].items()})
                        
                        # Handle timing info if present
                        if "timing" in batch.meta_info:
                            timing_raw.update(batch.meta_info["timing"])
                            batch.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # Add UIDs after batch is populated with actual data
                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

                    batch.batch["response_mask"] = batch.batch["loss_mask"]
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if do_profile:
                    self.actor_rollout_wg.stop_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.stop_profile()
                    if self.use_critic:
                        self.critic_wg.stop_profile()
                    if self.use_rm:
                        self.rm_wg.stop_profile()

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

    def _validate(self):
        """
        Validation method for AgentTrainer using multi-turn rollouts.
        Modified from original RayPPOTrainer._validate() to use SyncMultiTurnRollout.
        """
        from collections import defaultdict
        import time
        
        reward_tensor_lst = []
        data_source_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        # ─────────────────── MODIFICATION: Use rollout-based validation config ───────────────────
        env_metric_dict = {}
        for step in range(self.config.trainer.validation_steps):
            # ✅ MODIFICATION: Use agent_group_num * agent_group_size for total validation agents
            agent_group_num = self.config.rollout.agent_group_num
            agent_group_size = self.config.rollout.agent_group_size
            total_validation_agents = agent_group_num * agent_group_size
            
            print(f"Validation step {step+1}: Running {total_validation_agents} agents ({agent_group_num} groups × {agent_group_size} agents/group)")
            
            input_texts = ["" for _ in range(total_validation_agents)]
            sample_inputs.extend(input_texts)
            
            meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            test_gen_batch = DataProto(batch=None, non_tensor_batch=None, meta_info=meta_info)
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # ─────────────────── MODIFICATION: Use multi-turn rollout with rollout() + build_ppo_batch() ───────────────────
            start_time = time.time()
            
            # Initialize multi-turn rollout if not already done
            if self.multi_turn_rollout is None:
                self.init_multi_turn_rollout()
            
            # Type narrowing assertion - we know it's not None after initialization  
            assert self.multi_turn_rollout is not None, "multi_turn_rollout should be initialized"
            
            # ✅ MODIFICATION: Call rollout() first, then build_ppo_batch() 
            final_env_outs = self.multi_turn_rollout.rollout()
            test_batch = self.multi_turn_rollout.build_ppo_batch()
            
            end_time = time.time()
            print(f"validation generation time: {end_time - start_time} seconds")
            
            # ✅ MODIFICATION: Use "val-env/" prefix for environment metrics
            for key, value in test_batch.meta_info["metrics"].items():
                if "val-env/" + key not in env_metric_dict:
                    env_metric_dict["val-env/" + key] = []
                env_metric_dict["val-env/" + key].append(value)

            # Store generated outputs
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            # ─────────────────── ORIGINAL: Evaluate using reward function ───────────────────
            # evaluate using reward_function
            reward_tensor = self.val_reward_fn(test_batch)
            
            # ✅ MODIFICATION: Store scores and collect for final processing
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        # ✅ MODIFICATION: Process tensor list and data sources like the example
        reward_tensor_lst = [i.sum(-1).cpu() for i in reward_tensor_lst]
        reward_tensor = torch.cat(reward_tensor_lst)  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # ✅ MODIFICATION: Evaluate test_score based on data source like the example
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        # ✅ MODIFICATION: Simple metric processing with fallback
        try:
            from verl.utils.metric import reduce_metrics
        except ImportError:
            # Fallback implementation
            def reduce_metrics(metrics_dict):
                return {k: np.mean(v) if isinstance(v, list) else v for k, v in metrics_dict.items()}

        metric_dict = reduce_metrics(env_metric_dict)
        
        # ✅ MODIFICATION: Add test scores per data source like the example
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val-env/test_score/{data_source}'] = np.mean(rewards)

        return metric_dict