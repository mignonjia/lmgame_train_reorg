# sync_multi_turn_rollout.py
from typing import List, Dict, Any, Union
import torch
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from agents import get_agent_cls, REGISTERED_AGENTS


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
        
        # Determine batch size
        self.n_agents = getattr(cfg, "agent_length", getattr(cfg, "env_length", 1))
        
        # Initialize agent configuration from config
        self._setup_agent_config()
        
        # Initialize agents and tracking structures
        self._init_batch_agents()
        
        # Global turn counter
        self.step_cnt = 0

    def _setup_agent_config(self):
        """
        Setup agent configuration, supporting both single-agent and mixed-agent setups.
        Agent class is always resolved from config.
        """
        # Extract agent configuration from config
        agent_config = getattr(self.cfg, 'agent', {})
        
        if self._is_mixed_agent_config(agent_config):
            # Mixed agent training
            self.env_mixture = agent_config['env_mixture']
            self.is_mixed_agent = True
            self.agent_cls = None  # Will be determined per agent
        else:
            # Single agent training
            agent_name = agent_config.get('agent_name', 'sokobanAgent')
            self.agent_cls = get_agent_cls(agent_name)
            self.is_mixed_agent = False

    def _is_mixed_agent_config(self, agent_config) -> bool:
        """Check if config specifies mixed agent training."""
        return 'env_mixture' in agent_config and isinstance(agent_config.env_mixture, list)

    def _init_batch_agents(self):
        """
        Build self.agents: Dict[int, Agent], self.agent_ids, self.done_mask, self.env_outs
        Each agent handles its own history & recorder.
        """
        self.agent_ids = list(range(self.n_agents))
        
        if self.is_mixed_agent:
            self.agents = self._create_mixed_agents()
        else:
            self.agents = self._create_single_type_agents()
        
        # Tracking structures (indexed by position, not agent_id)
        self.done_mask = torch.zeros(self.n_agents, dtype=torch.bool)
        self.env_outs = [
            self.agents[agent_id].get_initial_env_outputs() 
            for agent_id in self.agent_ids
        ]

    def _create_single_type_agents(self) -> Dict[int, Any]:
        """Create agents of a single type."""
        if self.agent_cls is None:
            raise ValueError("agent_cls is None but trying to create single type agents")
        return {
            agent_id: self.agent_cls(
                self.cfg.env_template[agent_id] if hasattr(self.cfg, "env_template") 
                else self.cfg
            )
            for agent_id in self.agent_ids
        }

    def _create_mixed_agents(self) -> Dict[int, Any]:
        """Create mixed agents based on env_mixture configuration."""
        # Calculate distribution
        agents = {}
        mixture_specs = self.env_mixture
        
        # Normalize ratios
        total_ratio = sum(spec['ratio'] for spec in mixture_specs)
        normalized_ratios = [spec['ratio'] / total_ratio for spec in mixture_specs]
        
        # Assign agents based on ratios
        agent_idx = 0
        for i, spec in enumerate(mixture_specs):
            agent_cls = get_agent_cls(spec['agent_name'])
            count = int(normalized_ratios[i] * self.n_agents)
            
            # Handle rounding for last group
            if i == len(mixture_specs) - 1:
                count = self.n_agents - agent_idx
            
            for _ in range(count):
                if agent_idx >= self.n_agents:
                    break
                    
                config = (self.cfg.env_template[agent_idx] if hasattr(self.cfg, "env_template") 
                         else self.cfg)
                agents[agent_idx] = agent_cls(config)
                agent_idx += 1
                
        return agents

    # ─────────────────── UTILITY METHODS ───────────────────
    def get_agent_distribution(self) -> Dict[str, int]:
        """Get distribution of agent types in current batch."""
        distribution = {}
        for agent in self.agents.values():
            agent_type = agent.__class__.__name__
            distribution[agent_type] = distribution.get(agent_type, 0) + 1
        return distribution

    def get_agents_by_type(self, agent_type: str) -> List[int]:
        """Get agent IDs of a specific type."""
        return [agent_id for agent_id, agent in self.agents.items() 
                if agent.__class__.__name__ == agent_type]

    # ─────────────────── PROMPT COLLECTION ───────────────────
    def _collect_prompts(self):
        """
        Method 3: For every alive agent, call agent.get_llm_prompts(env_out);
        pad to GPU multiple; return DataProto, idx_map.
        No recorder logic here.
        """
        rows, idx_map = [], []
        
        for idx, agent_id in enumerate(self.agent_ids):
            if self.done_mask[idx]:
                continue
                
            agent = self.agents[agent_id]
            env_out = self.env_outs[idx]
            rows.append(agent.get_llm_prompts(env_out))
            idx_map.append(idx)
        
        if not rows:  # All agents done
            return None, None
        
        # Pad to GPU multiple
        while len(rows) % self.cfg.n_gpus_per_node:
            rows.append(rows[-1].copy())
        
        return DataProto.from_single_dict(collate_fn(rows)), idx_map

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

    # ─────────────────── REPLY APPLICATION ───────────────────
    def _apply_replies(self, replies, idx_map):
        """
        Method 5: Pass text back to the correct agent via agent.get_env_outputs(reply);
        update self.done_mask, self.env_outs.
        Agents take care of update_history() internally.
        """
        for reply, row_idx in zip(replies, idx_map):
            if self.done_mask[row_idx]:
                continue
                
            agent_id = self.agent_ids[row_idx]
            agent = self.agents[agent_id]
            
            # Agent handles history updates internally
            self.env_outs[row_idx] = agent.get_env_outputs(reply)
            self.done_mask[row_idx] = self.env_outs[row_idx].done

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
            self._apply_replies(replies, idx_map)
            
            self.step_cnt += 1
        
        return self.env_outs

    # ─────────────────── PPO BATCH BUILDING ───────────────────
    def build_update_batch(self):
        """
        Method 7: After rollout, ask every agent for a single make_update_row() dict;
        collate_fn → DataProto. Ready for PPO update.
        """
        rows = [
            self.agents[agent_id].make_update_row() 
            for agent_id in self.agent_ids
        ]
        return DataProto.from_single_dict(collate_fn(rows))

    # ─────────────────── LIFECYCLE MANAGEMENT ───────────────────
    def reset(self):
        """
        Method 8 (optional): If your trainer calls it between epochs,
        loop over agents, agent.reset(), refresh masks/obs.
        Only needed if the outer loop expects it.
        """
        for agent_id in self.agent_ids:
            self.agents[agent_id].reset()
        
        # Reset tracking structures
        self.done_mask = torch.zeros(self.n_agents, dtype=torch.bool)
        self.env_outs = [
            self.agents[agent_id].get_initial_env_outputs() 
            for agent_id in self.agent_ids
        ]
        self.step_cnt = 0

    def close(self):
        """
        Method 9 (optional): agent.close() and/or env.close() for tidy teardown.
        """
        for agent_id in self.agent_ids:
            agent = self.agents[agent_id]
            if hasattr(agent, 'close'):
                agent.close()
            # If agent has separate env reference
            if hasattr(agent, 'env') and hasattr(agent.env, 'close'):
                agent.env.close()
