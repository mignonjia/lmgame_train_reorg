# agent_trainer.py
from typing import Dict, Any
import torch
import numpy as np
import uuid
from copy import deepcopy
from pprint import pprint
from tqdm import tqdm

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
        class DummyDataset:
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
            gen_batch: DataProto containing prompts for generation (mostly metadata)
            
        Returns:
            DataProto: Generated sequences in format compatible with PPO pipeline
        """
        # Initialize multi-turn rollout if not already done
        if self.multi_turn_rollout is None:
            self.init_multi_turn_rollout()
        
        # Run multi-turn rollout to get complete trajectories
        assert self.multi_turn_rollout is not None  # Type narrowing for linter
        final_env_outs = self.multi_turn_rollout.rollout()
        
        # Build update batch containing full trajectories and rewards
        # This already returns a complete DataProto with all necessary fields
        rollout_batch = self.multi_turn_rollout.build_ppo_batch()
        
        # Return the complete DataProto directly - no need to rebuild it
        return rollout_batch

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
                        gen_batch_output = self._generate_multi_turn_sequences(gen_batch)
                        
                        # Handle timing info if present
                        if "timing" in gen_batch_output.meta_info:
                            timing_raw.update(gen_batch_output.meta_info["timing"])
                            gen_batch_output.meta_info.pop("timing", None)

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

                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    batch.batch["response_mask"] = compute_response_mask(batch)
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
        
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        # ─────────────────── MODIFICATION: Use agent-based validation ───────────────────
        env_metric_dict = {}
        for step in range(self.config.trainer.validation_steps):
            # Store original inputs (empty for agent-based validation)
            input_texts = ["" for _ in range(self.config.es_manager.val.env_groups * self.config.es_manager.val.group_size)]
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

            # ─────────────────── MODIFICATION: Use multi-turn rollout instead of single generation ───────────────────
            import time
            start_time = time.time()
            
            # Initialize multi-turn rollout if not already done
            if self.multi_turn_rollout is None:
                self.init_multi_turn_rollout()
            
            # Run multi-turn rollout for validation
            final_env_outs = self.multi_turn_rollout.rollout()
            test_batch = self.multi_turn_rollout.build_ppo_batch()
            
            end_time = time.time()
            print(f"validation generation time: {end_time - start_time} seconds")
            
            # Collect environment metrics from rollout
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
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        # ─────────────────── ORIGINAL: Log and dump generations ───────────────────
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        # ─────────────────── MODIFICATION: Process validation metrics with env metrics ───────────────────
        try:
            from verl.trainer.ppo.metric_utils import process_validation_metrics
            from verl.utils.metric import reduce_metrics
        except ImportError:
            # Fallback implementations if verl is not available
            def process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict):
                return {"unknown": {"reward": {"mean@1": np.mean(reward_extra_infos_dict.get("reward", [0]))}}}
            
            def reduce_metrics(metrics_dict):
                return {k: np.mean(v) if isinstance(v, list) else v for k, v in metrics_dict.items()}
        
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = reduce_metrics(env_metric_dict)

        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict