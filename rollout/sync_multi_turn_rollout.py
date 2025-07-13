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
        Agent class is resolved from config and agent config is extracted.
        """
        # Get agent name from train list (first item)
        train_agents = getattr(self.cfg, 'train', ['sokobanAgent'])
        agent_name = train_agents[0] if train_agents else 'sokobanAgent'
        
        # Resolve agent class from registry
        self.agent_cls = get_agent_cls(agent_name)
        
        # Extract agent configuration from agents.yaml format
        # The config should contain the agent configuration under the agent name
        if hasattr(self.cfg, agent_name):
            self.agent_config = getattr(self.cfg, agent_name)
        elif agent_name in self.cfg:
            self.agent_config = self.cfg[agent_name]
        else:
            raise ValueError(f"Agent configuration for '{agent_name}' not found in config")

    def _init_batch_agents(self):
        """
        Build self.agents: List[Agent] and collect batch of agents.reset() outputs.
        Each agent handles its own history & recorder.
        Agents are grouped based on agent_group_size for training purposes.
        Initializes self.done_mask and self.env_outs with initial environment states.
        """
        # Create agents of the same type
        if self.agent_cls is None:
            raise ValueError("agent_cls is None but trying to create agents")
        
        # Get group size from config, default to 1 if not specified
        agent_group_size = getattr(self.cfg, 'agent_group_size', 1)
        
        # Calculate number of groups
        num_groups = self.n_agents // agent_group_size
        if self.n_agents % agent_group_size != 0:
            raise ValueError(f"agent_batch_size ({self.n_agents}) must be divisible by agent_group_size ({agent_group_size})")
        
        print(f"Creating {self.n_agents} agents in {num_groups} groups of size {agent_group_size}")
        
        self.agents = []
        initial_env_outs = []
        
        for idx in range(self.n_agents):
            # Calculate group_id for this agent
            group_id = idx // agent_group_size
            
            # Create agent with the extracted agent configuration
            agent = self.agent_cls(
                config=self.agent_config,
                agent_id=idx,
                group_id=group_id
            )
            
            # Reset agent and collect initial environment output
            initial_env_out = agent.reset()
            
            self.agents.append(agent)
            initial_env_outs.append(initial_env_out)
            print(f"  Agent {idx}: group_id={group_id}, initial_state_preview={initial_env_out.state[:20]}...")
        
        # Initialize tracking structures with batch of reset outputs
        self.done_mask = torch.zeros(self.n_agents, dtype=torch.bool)
        self.env_outs = initial_env_outs

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
            prompt_str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Add answer format prompt
            enable_think = getattr(self.cfg.agent, 'enable_think', False)
            if enable_think:
                prompt_str += "<think>"  # Force LLM to think before answering
            else:
                prompt_str += "<answer>"  # Force LLM to answer
            
            llm_input_texts.append(prompt_str)
        
        # Tokenize all prompts at once
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
        
        # Create DataProto
        batch_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": input_ids[:, 1:],  # Remove the first token
        }
        
        return DataProto.from_single_dict(batch_dict)



    # ─────────────────── MAIN ROLLOUT LOOP ───────────────────
    def rollout(self):
        """
        Main rollout loop using batch LLM prompts approach.
        Iterate cfg.agent.max_turn turns, breaking early if all done.
        """
        for turn in range(self.cfg.agent.max_turn):
            if self.done_mask.all():
                break

            # Generate batch of LLM prompts from current env outputs
            batch_prompts = self.get_batch_llm_prompts(self.env_outs)
            
            # Generate responses using batch dispatch
            lm_outputs = self.actor_wg.generate_sequences(batch_prompts)
            
            # Decode responses
            replies = self.tokenizer.batch_decode(
                lm_outputs.batch["responses"], 
                skip_special_tokens=True
            )
            
            # Update environment outputs for all agents
            for idx, reply in enumerate(replies):
                if self.done_mask[idx]:
                    continue
                    
                agent = self.agents[idx]
                # Agent handles history updates internally
                self.env_outs[idx] = agent.get_env_outputs(reply)
                self.done_mask[idx] = self.env_outs[idx].done

            #TODO: Early stopping. If max_actions_all_turns is reached or env is done, break.
            
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
