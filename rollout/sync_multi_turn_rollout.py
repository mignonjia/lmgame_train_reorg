# sync_multi_turn_rollout.py
from typing import List, Dict, Any, Union
import torch
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
import verl.utils.torch_functional as verl_F
from agents import get_agent_cls, REGISTERED_AGENTS
import numpy as np
from tensordict import TensorDict


class SyncMultiTurnRollout:
    """
    Vectorised, synchronous, multi-turn rollout manager.
    Batch size = cfg.agent_length (fallback to cfg.env_length).
    
    Each agent manages its own environment, history, and recorder internally.
    This class orchestrates the batch processing and LLM calls.
    """

    # ─────────────────── INITIALIZATION ───────────────────
    def __init__(self, actor_rollout_wg, cfg, tokenizer, processor):
        """
        Initialize rollout manager. Agent class is resolved from config.
        
        Args:
            actor_rollout_wg: Actor rollout worker group
            cfg: Configuration object
            tokenizer: Tokenizer for text processing
            processor: Data processor
        """
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.processor = processor
        self.actor_wg = actor_rollout_wg
        
        # ✅ MODIFICATION: Calculate total agents from agent_group_num * agent_group_size
        agent_group_num = getattr(cfg.rollout, "agent_group_num", 1)
        agent_group_size = getattr(cfg.rollout, "agent_group_size", 1)
        self.n_agents = agent_group_num * agent_group_size
        self.agent_group_num = agent_group_num
        self.agent_group_size = agent_group_size
        

        
        # Initialize agent configuration from config
        self._setup_agent_config()

        self._init_batch_agents()
        
        # Global turn counter
        self.step_cnt = 0

    def _setup_agent_config(self):
        """
        Setup agent configuration for single agent type.
        Agent class is resolved from config and agent config is extracted.
        """
        # Get agent name from rollout config train list (first item)
        train_agents = getattr(self.cfg.rollout, 'train', ['sokobanAgent'])
        agent_name = train_agents[0] if train_agents else 'sokobanAgent'
        
        # Resolve agent class from registry
        self.agent_cls = get_agent_cls(agent_name)
        
        # Extract agent configuration from agents.yaml format
        # The config should contain the agent configuration under the agent name
        self.agent_config = self.cfg[agent_name]
        
        # Get max_turns from agent config
        self.max_turns = self.agent_config['agent_config'].get('max_turns', 5)

    def _init_batch_agents(self):
        """
        Build self.agents: List[Agent] without resetting them.
        Each agent handles its own history & recorder.
        Agents are grouped based on agent_group_size for training purposes.
        Actual reset happens in rollout() via _reset_batch_agents().
        """
        # Create agents of the same type
        if self.agent_cls is None:
            raise ValueError("agent_cls is None but trying to create agents")
        
        print(f"Creating {self.n_agents} agents in {self.agent_group_num} groups of size {self.agent_group_size}")
        
        # Verify the math
        if self.n_agents != self.agent_group_num * self.agent_group_size:
            raise ValueError(f"Total agents ({self.n_agents}) != agent_group_num ({self.agent_group_num}) × agent_group_size ({self.agent_group_size})")
        
        self.agents = []
        
        for idx in range(self.n_agents):
            group_id = idx // self.agent_group_size
            
            # Create agent with the extracted agent configuration
            agent = self.agent_cls(
                config=self.agent_config,
                agent_id=idx,
                group_id=group_id
            )
            
            self.agents.append(agent)
        
        # Initialize tracking structures - actual env_outs will be set in rollout()
        self.done_mask = torch.zeros(self.n_agents, dtype=torch.bool)
        self.env_outs = None  # Will be initialized in _reset_batch_agents()

    # ─────────────────── BATCH LLM PROMPTS ───────────────────
    def get_batch_llm_prompts(self, env_outputs):
        """
        Generate batch of LLM prompts from environment outputs.
        Each agent.get_llm_prompts(env_out) returns messages format.
        
        Args:
            env_outputs: List of environment outputs from agents
            
        Returns:
            DataProto: Batched DataProto containing input_ids, attention_mask, position_ids
        """
        llm_input_texts = []
        
        for idx, env_out in enumerate(env_outputs):
            if self.done_mask[idx]:
                # For done agents, use empty prompt
                llm_input_texts.append("")
                continue
                
            agent = self.agents[idx]
            
            # Each agent returns messages format
            messages = agent.get_llm_prompts(env_out)
            
            # Apply chat template to convert messages to text
            try:
                prompt_str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception as e:
                prompt_str = "System error in chat template"
            
            # Add answer format prompt based on agent's enable_think setting
            agent = self.agents[idx]
        
            
            llm_input_texts.append(prompt_str)
        

        
        # Check prompt statistics
        empty_prompts = sum(1 for p in llm_input_texts if not p or len(p.strip()) == 0)
        total_length = sum(len(p) for p in llm_input_texts)
        print(f"   Empty prompts: {empty_prompts}/{len(llm_input_texts)}")
        print(f"   Total character count: {total_length}")
        print(f"   Average length: {total_length/len(llm_input_texts) if llm_input_texts else 0:.1f}")
        
        # Tokenize all prompts using verl_F for more universal processing
        batch_list = []
        
        for i, prompt_str in enumerate(llm_input_texts):
            
            if not prompt_str or len(prompt_str.strip()) == 0:
                prompt_str = "Please respond."  # Fallback
            
            # Use verl_F.tokenize_and_postprocess_data for consistent processing
            try:
                input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                    prompt=prompt_str,
                    tokenizer=self.tokenizer,
                    max_length=self.cfg.max_prompt_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True,  # Left pad for batch generation
                    truncation=self.cfg.rollout.truncation
                )
                
                
                # Check for all-padding sequences
                pad_token_id = self.tokenizer.pad_token_id
                non_pad_count = (input_ids != pad_token_id).sum().item()
                total_tokens = input_ids.numel()
                
            except Exception as e:
                # Create emergency fallback tokens
                fallback_text = "Please respond."
                input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                    prompt=fallback_text,
                    tokenizer=self.tokenizer,
                    max_length=self.cfg.max_prompt_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True,
                    truncation=self.cfg.rollout.truncation
                )
            
            # Compute position ids
            from verl.utils.model import compute_position_id_with_mask
            position_ids = compute_position_id_with_mask(attention_mask)
            
            # Build row_dict for each prompt
            row_dict = {
                'input_ids': input_ids.squeeze(0),  # Remove batch dimension
                'attention_mask': attention_mask.squeeze(0),
                'position_ids': position_ids.squeeze(0),
                'responses': input_ids.squeeze(0)[1:],  # Remove first token for responses
            }
            batch_list.append(row_dict)
        
        # Use collate_fn to batch the data, then convert to DataProto
        try:
            batch_dict = collate_fn(batch_list)
            result_dataproto = DataProto.from_single_dict(batch_dict)
            return result_dataproto
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise

    # ─────────────────── BATCH ENV OUTPUTS ───────────────────
    def get_batch_env_outputs(self, lm_outputs):
        """
        Process LLM outputs and update environment outputs for all agents.
        
        Args:
            lm_outputs: DataProto containing LLM responses
            
        Returns:
            List: Updated environment outputs from all agents
        """
        # Ensure env_outs is initialized (should be done by _reset_batch_agents)
        if self.env_outs is None:
            raise RuntimeError("env_outs not initialized. Call rollout() or _reset_batch_agents() first.")
        
        # Decode responses
        replies = self.tokenizer.batch_decode(
            lm_outputs.batch["responses"], 
            skip_special_tokens=True
        )

        # Update environment outputs for all agents
        updated_env_outs = []
        
        for idx, reply in enumerate(replies):
            if self.done_mask[idx]:
                # Keep existing env output for done agents
                updated_env_outs.append(self.env_outs[idx])
                continue
                
            agent = self.agents[idx]
            # Agent handles history updates internally
            env_out = agent.get_env_outputs(reply)
            updated_env_outs.append(env_out)
            
            # Update tracking structures
            self.env_outs[idx] = env_out
            self.done_mask[idx] = env_out.done
        
        return updated_env_outs

    # ─────────────────── LLM GENERATION ───────────────────
    def generate_sequences(self, lm_inputs: DataProto):
        """
        Generate sequences using the actor worker group.
        
        Args:
            lm_inputs: DataProto containing input_ids, attention_mask, position_ids
            
        Returns:
            DataProto: Generated sequences
        """
        # TODO: add kv cache both for the vllm wrapper here and for verl vllm.
        try:
            from verl.trainer.ppo.ray_trainer import RayWorkerGroup
            from verl.utils.dataset.rl_dataset import pad_dataproto_to_divisor, unpad_dataproto
        except ImportError:
            RayWorkerGroup = None
            pad_dataproto_to_divisor = None
            unpad_dataproto = None
        
        if RayWorkerGroup is not None and isinstance(self.actor_wg, RayWorkerGroup):
            padded_lm_inputs, pad_size = pad_dataproto_to_divisor(lm_inputs, self.actor_wg.world_size)
            padded_lm_outputs = self.actor_wg.generate_sequences(padded_lm_inputs)
            lm_outputs = unpad_dataproto(padded_lm_outputs, pad_size=pad_size)
            lm_outputs.meta_info = lm_inputs.meta_info
            lm_outputs.non_tensor_batch = lm_inputs.non_tensor_batch
        else:
            lm_outputs = self.actor_wg.generate_sequences(lm_inputs)
        
        return lm_outputs

    # ─────────────────── MAIN ROLLOUT LOOP ───────────────────
    def rollout(self):
        """
        Main rollout loop using batch LLM prompts approach.
        Iterate cfg.agent.max_turn turns, breaking early if all done.
        """
        self._reset_batch_agents()
        
        for turn in range(self.max_turns):
            if self.done_mask.all():
                break

            # Generate batch of LLM prompts from current env outputs
            batch_prompts = self.get_batch_llm_prompts(self.env_outs)
            
            # Generate responses using batch dispatch
            lm_outputs = self.generate_sequences(batch_prompts)
            
            # Process LLM outputs and update environment outputs
            self.env_outs = self.get_batch_env_outputs(lm_outputs)

            #TODO: Early stopping. If max_actions_all_turns is reached or env is done, break.
            
            self.step_cnt += 1

        return self.env_outs

    # ─────────────────── PPO BATCH BUILDING ───────────────────
    def _collect_final_rollout_states(self) -> List[Dict]:
        """
        Collect final rollout states from all agents.
        
        Returns:
            List[Dict]: List of rollout state dictionaries from all agents
        """
        env_outputs = []
        for idx in range(self.n_agents):
            agent = self.agents[idx]
            rollout_state = agent.get_final_rollout_states()
            env_outputs.append(rollout_state)
        return env_outputs

    def build_ppo_batch(self) -> DataProto:
        """
        Build PPO batch from the final batch rollout states.
        Converts collected rollout states to DataProto format for PPO training.
        """
        # Step 1: Collect final rollout states from all agents
        env_outputs = self._collect_final_rollout_states()
        
        # Step 2: Convert to DataProto format (similar to get_lm_inputs with prepare_for_update=True)
        llm_input_texts = []
        messages_list = []
        
        for env_output in env_outputs:
            # Build messages from trajectory history
            system_prompt = self.agent_config.get('system_prompt', "You are a helpful AI assistant that solves Sokoban puzzles step by step.")
            prompt = self.agent_config.get('prompt', "You are solving the Sokoban puzzle.")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Process each turn in the history
            for idx, content in enumerate(env_output["history"]):
                messages[-1]["content"] += f"\nTurn {idx + 1}:\n"
                
                if "state" in content:
                    # Use the first agent's enable_think setting (all agents of same type should have same setting)
                    enable_think = getattr(self.agents[0], 'enable_think', True) if self.agents else True
                    FORMAT_PROMPT = "<think> [Your thoughts] </think> <answer> [your answer] </answer>" if enable_think else "<answer> [your answer] </answer>"
                    messages[-1]["content"] += f"State:\n{content['state']}\nYou have {content['actions_left']} actions left. Always output: {FORMAT_PROMPT} with no extra text. Strictly follow this format.\n"
                
                if "llm_response" in content:
                    messages.append({"role": "assistant", "content": content["llm_response"]})
                
                if "reward" in content:
                    messages.append({"role": "user", "content": f"Reward:\n{content['reward']}\n"})
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
            llm_input_texts.append(text)
            messages_list.append(messages)
        
        # Tokenize all texts
        inputs = self.tokenizer(
            llm_input_texts, 
            return_tensors="pt", 
            padding=True, 
            padding_side="left", 
            truncation=False
        )
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        
        # Compute position ids
        from verl.utils.model import compute_position_id_with_mask
        position_ids = compute_position_id_with_mask(attention_mask)
        
        # Extract scores from trajectory history
        scores = []
        for env_output in env_outputs:
            trajectory_scores = [entry.get('reward', 0.0) for entry in env_output['history']]
            scores.append(trajectory_scores)
        
        # Get masks and scores (assuming this function exists in verl)
        try:
            from verl.utils.reward_score import get_masks_and_scores
            score_tensor, loss_mask, response_mask = get_masks_and_scores(
                input_ids, 
                self.tokenizer, 
                scores, 
                use_turn_scores=getattr(self.cfg.rollout, 'use_turn_scores', False),
                enable_response_mask=getattr(self.cfg.rollout, 'enable_response_mask', True)
            )
        except ImportError:
            # Fallback if function not available
            score_tensor = torch.zeros_like(input_ids[:, 1:], dtype=torch.float)
            loss_mask = torch.ones_like(input_ids[:, 1:], dtype=torch.float)
            response_mask = torch.ones_like(input_ids[:, 1:], dtype=torch.float)
        
        # Build DataProto with proper TensorDict
        llm_inputs = DataProto()
        
        # Ensure all tensors are on the same device and have correct shapes
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # Create TensorDict with proper batch_size parameter
        final_responses = input_ids[:, 1:]  # remove the first token
        llm_inputs.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": final_responses,
            "loss_mask": loss_mask,
            "rm_scores": score_tensor,
            "original_rm_scores": score_tensor,
        }, batch_size=batch_size, device=device)
        
        # Non-tensor batch data
        llm_inputs.non_tensor_batch = {
            "env_ids": np.array([env_output["env_id"] for env_output in env_outputs], dtype=object),
            "group_ids": np.array([env_output["group_id"] for env_output in env_outputs], dtype=object),
            "messages_list": np.array(messages_list, dtype=object),
        }
        
        # Collect metrics
        metrics = {}
        for env_output in env_outputs:
            for key, value in env_output["metrics"].items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)
        
        # Calculate mean metrics
        mean_metrics = {}
        for key, values in metrics.items():
            if isinstance(values, list):
                mean_metrics[key] = np.mean(values)
            else:
                mean_metrics[key] = values
        
        # Add response length metric
        if response_mask is not None:
            mean_metrics["response_length"] = response_mask.sum(dim=-1).float().mean().item()
        
        llm_inputs.meta_info = {"metrics": mean_metrics}
        
        return llm_inputs

    # ─────────────────── LIFECYCLE MANAGEMENT ───────────────────

    def _reset_batch_agents(self, seed=None):
        """
        Reset all agents and collect the batch of initial env outputs.
        This function resets the rollout manager for new epoch/rollout.
        
        Args:
            seed: Optional base seed for reproducibility. If None, generates random seed.
        """
        import random
        
        # Generate base seed following reference implementation pattern
        if seed is None:
            # Generate random seed for training, consistent seed for validation  
            base_seed = random.randint(0, 1000000)
        else:
            base_seed = seed
            
        # Generate group seeds: agents within same group share environment
        # Different groups get different environments
        group_seeds = [base_seed + group_id for group_id in range(self.agent_group_num)]
        
        initial_env_outs = []
        
        for idx in range(self.n_agents):
            agent = self.agents[idx]
            group_id = idx // self.agent_group_size
            
            # All agents in the same group use the same seed (same environment)
            group_seed = group_seeds[group_id]
            
         
            
            # Reset agent with group-specific seed
            initial_env_out = agent.reset(seed=group_seed)
            initial_env_outs.append(initial_env_out)
        
        # Update tracking structures with batch of reset outputs
        self.done_mask = torch.zeros(self.n_agents, dtype=torch.bool)
        self.env_outs = initial_env_outs
        self.step_cnt = 0

        
    def reset(self, seed=None):
        """
        Public reset method for external use (e.g., called by trainer between epochs).
        Delegates to _reset_batch_agents() for actual reset logic.
        
        Args:
            seed: Optional base seed for reproducibility
        """
        self._reset_batch_agents(seed=seed)

    def close(self):
        """
        agent.close() and/or env.close() for tidy teardown.
        """
        for idx in range(self.n_agents):
            agent = self.agents[idx]
            if hasattr(agent, 'close'):
                agent.close()
            # If agent has separate env reference
            if hasattr(agent, 'env') and hasattr(agent.env, 'close'):
                agent.env.close()
