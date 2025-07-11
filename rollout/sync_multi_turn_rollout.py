# sync_multi_turn_rollout.py
from typing import List, Dict, Any, Union
import torch
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
import verl.utils.torch_functional as verl_F
from agents import get_agent_cls, REGISTERED_AGENTS
import numpy as np


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
        
        # Determine batch size from agent_batch_size
        self.n_agents = getattr(cfg, "agent_batch_size", 1)
        
        # Initialize agent configuration from config
        self._setup_agent_config()
        
        # Initialize agents and tracking structures
        self._init_batch_agents()
        
        # Global turn counter
        self.step_cnt = 0

    def _setup_agent_config(self):
        """
        Setup agent configuration for single agent type.
        Agent class is resolved from config.
        """
        # Get agent name from train list (first item)
        train_agents = getattr(self.cfg, 'train', ['sokobanAgent'])
        agent_name = train_agents[0] if train_agents else 'sokobanAgent'
        
        # Resolve agent class from registry
        self.agent_cls = get_agent_cls(agent_name)

    def _init_batch_agents(self):
        """
        Build self.agents: List[Agent], self.done_mask, self.env_outs
        Each agent handles its own history & recorder.
        """
        # Create agents of the same type
        if self.agent_cls is None:
            raise ValueError("agent_cls is None but trying to create agents")
        
        self.agents = [
            self.agent_cls(
                self.cfg.env_template[idx] if hasattr(self.cfg, "env_template") 
                else self.cfg
            )
            for idx in range(self.n_agents)
        ]
        
        # Tracking structures (indexed by position)
        self.done_mask = torch.zeros(self.n_agents, dtype=torch.bool)
        self.env_outs = [
            self.agents[idx].get_initial_env_outputs() 
            for idx in range(self.n_agents)
        ]

    # ─────────────────── PROMPT COLLECTION ───────────────────
    def _collect_prompts_from_agents(self):
        """
        Collect prompts from all alive agents.
        Each agent.get_llm_prompts(env_out) returns a single prompt string.
        
        Returns:
            prompts: List[str] - List of prompt strings from active agents
            idx_map: List[int] - Mapping from batch index to agent index
        """
        prompts, idx_map = [], []
        
        for idx in range(self.n_agents):
            if self.done_mask[idx]:
                continue
                
            agent = self.agents[idx]
            env_out = self.env_outs[idx]
            
            # Each agent returns a single prompt string
            prompt_str = agent.get_llm_prompts(env_out)
            prompts.append(prompt_str)
            idx_map.append(idx)
        
        if not prompts:  # All agents done
            return None, None
        
        # Pad to GPU multiple if needed
        while len(prompts) % self.cfg.n_gpus_per_node:
            # Duplicate the last prompt for padding
            prompts.append(prompts[-1])
        
        return prompts, idx_map

    def _convert_prompts_to_dataproto(self, prompts: List[str]):
        """
        Convert a batch of prompt strings to DataProto.
        Following the QwenVLRolloutManager pattern for text-only inputs.
        
        Args:
            prompts: List[str] - List of prompt strings
            
        Returns:
            DataProto: Batched DataProto containing input_ids, attention_mask, position_ids
        """
        batch_list = []
        
        for prompt_str in prompts:
            # Create row_dict for each prompt following QwenVLRolloutManager pattern
            row_dict = {}
            
            # Tokenize and postprocess the prompt
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_str,
                tokenizer=self.tokenizer,
                max_length=getattr(self.cfg, 'max_prompt_length', 2048),
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=False,
                truncation=getattr(self.cfg, 'truncation', 'right')
            )
            
            # Compute position ids
            from verl.utils.model import compute_position_id_with_mask
            position_ids = compute_position_id_with_mask(attention_mask)
            
            # Build row_dict
            row_dict['input_ids'] = input_ids.squeeze(0)  # Remove batch dimension
            row_dict['attention_mask'] = attention_mask.squeeze(0)  # Remove batch dimension  
            row_dict['position_ids'] = position_ids.squeeze(0)  # Remove batch dimension
            
            batch_list.append(row_dict)
        
        # Use collate_fn to batch the data, then convert to DataProto
        batch_dict = collate_fn(batch_list)
        return DataProto.from_single_dict(batch_dict)

    def _collect_prompts(self):
        """
        Main prompt collection function that orchestrates the two-step process:
        1. Collect prompts from all alive agents
        2. Convert the batch of prompts to DataProto
        
        Returns:
            batched_dataproto: DataProto containing batched prompts
            idx_map: List[int] - Mapping from batch index to agent index
        """
        # Step 1: Collect prompts from agents
        prompts, idx_map = self._collect_prompts_from_agents()
        if prompts is None:
            return None, None
        
        # Step 2: Convert batch of prompts to DataProto
        batched_dataproto = self._convert_prompts_to_dataproto(prompts)
        
        return batched_dataproto, idx_map

    # ─────────────────── LLM DISPATCH ───────────────────
    @torch.no_grad()
    def _dispatch(self, prompt_batch: DataProto):
        """
        Method 4: actor_rollout_wg.generate_sequences → List[str]
        Ultra-thin wrapper.
        """
        output = self.actor_wg.generate_sequences(prompt_batch)
        return self.tokenizer.batch_decode(
            output.batch["responses"], 
            skip_special_tokens=True
        )

    # ─────────────────── ENVIRONMENT OUTPUT UPDATES ───────────────────
    def update_env_outputs(self, replies, idx_map):
        """
        Pass LLM replies back to the correct agents and update environment outputs.
        Updates self.done_mask and self.env_outs based on agent responses.
        Agents take care of update_history() internally.
        
        Args:
            replies: List[str] - Generated text responses from LLM
            idx_map: List[int] - Mapping from batch index to agent index
        """
        for reply, idx in zip(replies, idx_map):
            if self.done_mask[idx]:
                continue
                
            agent = self.agents[idx]
            
            # Agent handles history updates internally
            self.env_outs[idx] = agent.get_env_outputs(reply)
            self.done_mask[idx] = self.env_outs[idx].done

    # ─────────────────── MAIN ROLLOUT LOOP ───────────────────
    def rollout(self):
        """
        Method 6: Iterate cfg.agent.max_turn turns, breaking early if all done;
        glue helpers 3→4→5. Returns final self.env_outs.
        """
        for turn in range(self.cfg.agent.max_turn):
            if self.done_mask.all():
                break
            
            # Collect prompts from active agents
            prompts, idx_map = self._collect_prompts()
            if prompts is None:
                break
            
            # Generate responses
            replies = self._dispatch(prompts)
            
            # Apply responses back to agents
            self.update_env_outputs(replies, idx_map)
            
            self.step_cnt += 1
        
        return self.env_outs

    # ─────────────────── PPO BATCH BUILDING ───────────────────
    def _collect_final_rollout_states(self):
        """
        Collect final rollout states from all agents.
        Each agent.get_final_rollout_states() returns a row_dict with trajectory data.
        
        Returns:
            row_dicts: List[Dict] - List of row dictionaries from agents
        """
        row_dicts = []
        
        for idx in range(self.n_agents):
            agent = self.agents[idx]
            
            # Each agent returns a complete row_dict with trajectory data
            row_dict = agent.get_final_rollout_states()
            row_dicts.append(row_dict)
        
        return row_dicts

    def _convert_rollout_states_to_dataproto(self, row_dicts: List[Dict]):
        """
        Convert a batch of rollout state dictionaries to DataProto.
        Following the same pattern as _convert_prompts_to_dataproto.
        Handles text-based rollout states from agents.
        
        Args:
            row_dicts: List[Dict] - List of row dictionaries containing trajectory data
                Each row_dict contains text fields that need tokenization
            
        Returns:
            DataProto: Batched DataProto containing all trajectory information
        """
        batch_list = []
        
        for row_dict in row_dicts:
            # Create tokenized row_dict for each rollout state
            tokenized_row_dict = {}
            
            # Handle the main conversation text (full trajectory)
            conversation_text = row_dict.get('conversation_history', row_dict.get('full_text', ''))
            if conversation_text:
                # Tokenize full conversation
                input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                    prompt=conversation_text,
                    tokenizer=self.tokenizer,
                    max_length=getattr(self.cfg, 'max_trajectory_length', 2048),
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=False,
                    truncation=getattr(self.cfg, 'truncation', 'right')
                )
                
                # Compute position ids
                from verl.utils.model import compute_position_id_with_mask
                position_ids = compute_position_id_with_mask(attention_mask)
                
                # Add core sequence data
                tokenized_row_dict['input_ids'] = input_ids.squeeze(0)
                tokenized_row_dict['attention_mask'] = attention_mask.squeeze(0)
                tokenized_row_dict['position_ids'] = position_ids.squeeze(0)
                tokenized_row_dict['sequences'] = input_ids.squeeze(0)  # Alias
                tokenized_row_dict['prompts'] = input_ids.squeeze(0)  # Alias
            
            # Handle final response text (for PPO log prob computation)
            final_response = row_dict.get('final_response', row_dict.get('response_text', ''))
            if final_response:
                # Tokenize final response
                response_ids, _ = verl_F.tokenize_and_postprocess_data(
                    prompt=final_response,
                    tokenizer=self.tokenizer,
                    max_length=getattr(self.cfg, 'max_response_length', 512),
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=False,
                    truncation=getattr(self.cfg, 'truncation', 'right')
                )
                
                tokenized_row_dict['responses'] = response_ids.squeeze(0)
            
            # Handle reward and scoring data (already tensors/numbers)
            for key in ['rm_scores', 'token_level_scores', 'reward', 'episode_reward']:
                if key in row_dict:
                    reward_value = row_dict[key]
                    if isinstance(reward_value, (int, float)):
                        # Convert scalar reward to token-level rewards
                        if 'responses' in tokenized_row_dict:
                            response_length = tokenized_row_dict['responses'].shape[0]
                            token_rewards = torch.zeros(response_length, dtype=torch.float)
                            token_rewards[-1] = reward_value  # Put reward at final token
                            tokenized_row_dict['rm_scores'] = token_rewards
                            tokenized_row_dict['token_level_scores'] = token_rewards
                    else:
                        # Already tensor format
                        tokenized_row_dict[key] = reward_value
            
            # Handle loss mask and other tensor fields
            for key in ['loss_mask', 'multi_turn_token_level_rewards', 'end_of_response_position_mask']:
                if key in row_dict:
                    tokenized_row_dict[key] = row_dict[key]
            
            # Handle metadata fields (non-tensor)
            for key in ['env_id', 'group_id', 'uid', 'metrics', 'tag', 'correct_answer']:
                if key in row_dict:
                    tokenized_row_dict[key] = row_dict[key]
            
            # Create default loss_mask if not provided (1 for assistant tokens)
            if 'loss_mask' not in tokenized_row_dict and 'attention_mask' in tokenized_row_dict:
                # Simple heuristic: assume second half of sequence is assistant response
                seq_len = tokenized_row_dict['attention_mask'].shape[0]
                loss_mask = torch.zeros_like(tokenized_row_dict['attention_mask'])
                if 'responses' in tokenized_row_dict:
                    response_len = tokenized_row_dict['responses'].shape[0]
                    loss_mask[-response_len:] = 1  # Mark response tokens for loss
                tokenized_row_dict['loss_mask'] = loss_mask
            
            batch_list.append(tokenized_row_dict)
        
        # Use collate_fn to batch the data, then convert to DataProto
        batch_dict = collate_fn(batch_list)
        return DataProto.from_single_dict(batch_dict)

    def build_update_batch(self):
        """
        Main function that orchestrates the two-step process:
        1. Collect final rollout states from all agents
        2. Convert the batch of rollout states to DataProto
        
        Returns:
            DataProto: Batch containing complete trajectory data ready for PPO training
        """
        # Step 1: Collect final rollout states from agents
        row_dicts = self._collect_final_rollout_states()
        
        # Step 2: Convert batch of rollout states to DataProto
        batched_dataproto = self._convert_rollout_states_to_dataproto(row_dicts)
        
        return batched_dataproto

    # ─────────────────── LIFECYCLE MANAGEMENT ───────────────────
    def reset(self):
        """
        If your trainer calls it between epochs,
        loop over agents, agent.reset(), refresh masks/obs.
        Only needed if the outer loop expects it.
        """
        for idx in range(self.n_agents):
            self.agents[idx].reset()
        
        # Reset tracking structures
        self.done_mask = torch.zeros(self.n_agents, dtype=torch.bool)
        self.env_outs = [
            self.agents[idx].get_initial_env_outputs() 
            for idx in range(self.n_agents)
        ]
        self.step_cnt = 0

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
