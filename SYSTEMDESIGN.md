# AgentTrainer: Multi-Turn PPO Training

## Overview

`AgentTrainer` inherits from `RayPPOTrainer` and replaces single-turn generation with multi-turn rollout logic using our `SyncMultiTurnRollout` class. The implementation is complete and functional with agent-based multi-turn interactions.

## Key Design Decisions

### 1. Inheritance Strategy
- **Inherit from RayPPOTrainer**: Maintains compatibility with existing VERL infrastructure
- **Minimal modifications**: Only replace the generation logic, keep everything else the same
- **Clean separation**: Multi-turn logic is encapsulated in `SyncMultiTurnRollout`

### 2. Agent Registration System
- **Agent Registry**: Agents are registered using `@register_agent("agentName")` decorator
- **Automatic Resolution**: Agent class is automatically resolved from config
- **Flexible Configuration**: Agent configs defined in YAML format

### 3. Main Modification Points

The implementation highlights three key modifications compared to the original `RayPPOTrainer`:

#### **1. Multi-turn Rollout Generation** (`_generate_multi_turn_sequences`)
```python
# ─────────────────── MODIFICATION: Multi-turn rollout generation replaces single-turn generation ───────────────────
def _generate_multi_turn_sequences(self, gen_batch: DataProto) -> tuple[DataProto, dict]:
    # Initialize multi-turn rollout if not already done
    if self.multi_turn_rollout is None:
        self.init_multi_turn_rollout()
    
    # Run multi-turn rollout to get complete trajectories
    self.multi_turn_rollout.rollout()
    
    # Build update batch containing full trajectories and rewards
    final_rollout_states = self.multi_turn_rollout._collect_final_rollout_states()
    rollout_batch = self.multi_turn_rollout.build_ppo_batch(final_rollout_states)
    
    rollout_batch, filter_metrics = self.multi_turn_rollout.filter_rollout(rollout_batch)
    
    return rollout_batch, filter_metrics
# ─────────────────── END MODIFICATION ───────────────────
```

#### **2. Multi-turn Validation** (`_validate`)
```python
# ─────────────────── MODIFICATION: Multi-turn rollout validation replaces val_dataloader validation ───────────────────
def _validate(self):
    # Uses multi-turn rollout for validation instead of val_dataloader
    # Calculates total_validation_agents = agent_group_num * agent_group_size
    # Runs self.multi_turn_rollout.rollout() for validation steps
    # Returns environment metrics with "val-env/" prefix
# ─────────────────── END MODIFICATION ───────────────────
```

#### **3. Main Training Loop Modification** (`fit`)
```python
# ─────────────────── MODIFICATION: Multi-turn rollout generation replaces actor_rollout_wg.generate_sequences ───────────────────
# Original: batch = self.actor_rollout_wg.generate_sequences(gen_batch)
# Modified: batch, rollout_metrics = self._generate_multi_turn_sequences(gen_batch)
# ─────────────────── END MODIFICATION ───────────────────
```

### 4. Integration Architecture

```
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   PPO Trainer   │───▶│   AgentTrainer      │───▶│ SyncMultiTurnRollout│
│   (Standard)    │    │   (Multi-turn)      │    │   (Agent Manager)   │
└─────────────────┘    └──────────────────────┘    └─────────────────────┘
                                │                              │
                                ▼                              ▼
                       ┌──────────────────────┐    ┌─────────────────────┐
                       │ _generate_multi_turn │    │  Agent[0..N]       │
                       │    _sequences()      │    │ (SokobanAgent etc.) │
                       └──────────────────────┘    └─────────────────────┘
```

## Implementation Status ✅

### Step 1: AgentTrainer Implementation ✅
```python
class AgentTrainer(RayPPOTrainer):
    def __init__(self, config, tokenizer, role_worker_mapping, 
                 resource_pool_manager, processor=None, **kwargs):
        # Agent class is automatically determined from config
        super().__init__(...)
        self.multi_turn_rollout = None  # Initialized later
```

### Step 2: SyncMultiTurnRollout Implementation ✅
```python
class SyncMultiTurnRollout:
    def __init__(self, actor_rollout_wg, cfg, tokenizer, processor):
        # Agent class resolved automatically from config
        self.agent_cls = get_agent_cls(agent_name)
        self.agents = [self.agent_cls(config=self.agent_config, 
                                     agent_id=idx, group_id=group_id)
                       for idx in range(self.n_agents)]
    
    def rollout(self):
        # Multi-turn rollout loop
        for turn in range(self.max_turns):
            if self.done_mask.all():
                break
            batch_prompts = self.get_batch_llm_prompts(self.env_outs)
            lm_outputs = self.generate_sequences(batch_prompts)
            self.env_outs = self.get_batch_env_outputs(lm_outputs)
```

### Step 3: Agent Implementation ✅
```python
@register_agent("sokobanAgent")
class SokobanAgent:
    def __init__(self, config, group_id=0, agent_id=0, seed=None, tag=None):
        # Agent configuration and environment setup
        
    def get_llm_prompts(self, env_out):
        # Convert environment outputs to LLM prompts
        
    def get_env_outputs(self, llm_response):
        # Process LLM outputs and get environment outputs
        
    def get_final_rollout_states(self):
        # Get final rollout states for PPO training
```

### Step 4: Training Script Modifications ✅
```python
# train.py modifications:
# ─────────────────── MODIFICATION: Import AgentTrainer instead of RayPPOTrainer ───────────────────
from trainer.agent_trainer import AgentTrainer

# ─────────────────── MODIFICATION: DummyRewardManager replaces load_reward_manager ───────────────────
class DummyRewardManager():
    # Custom reward manager implementation

# ─────────────────── MODIFICATION: Initialize AgentTrainer instead of RayPPOTrainer ───────────────────
trainer = AgentTrainer(
    config=config,
    tokenizer=tokenizer,
    # ... other arguments
    train_dataset=None,  # No datasets needed for agent-based training
    val_dataset=None,
)
```

## Configuration Requirements

The config should contain everything needed for:
1. **Standard PPO**: All existing RayPPOTrainer parameters
2. **Multi-turn rollout**: Agent configuration parameters
3. **Agent-specific settings**: Environment templates, max turns, etc.

Example config structure:
```yaml
# Standard PPO config
trainer:
  total_epochs: 10
  validation_steps: 10
  
algorithm:
  adv_estimator: "gae"

# Rollout config
rollout:
  train: ["sokobanAgent"]  # Agent to train
  agent_group_num: 2       # Number of agent groups
  agent_group_size: 2      # Agents per group
  use_turn_scores: false
  reward_normalization:
    grouping: "batch"
    method: "identity"
  rollout_filter_ratio: 1.0
  
# Agent-specific config
sokobanAgent:
  agent_config:
    max_turns: 5
    max_actions_all_turns: 10
    max_actions_per_turn: 5
    max_tokens: 100
    format_penalty: 0.0
    enable_think: true
    action_separator: "||"
    system_prompt: "You are a helpful AI assistant that solves Sokoban puzzles."
    prompt: "You are solving the Sokoban puzzle."
  env_config:
    # Environment-specific configuration
    grid_vocab:
      "@": "player"
      "#": "wall"
      "$": "box"
      ".": "target"
    action_lookup:
      0: "up"
      1: "down"
      2: "left"
      3: "right"
```

## Usage Example

```python
# The agent class is automatically resolved from config
# No need to explicitly pass agent_cls

# Initialize trainer
trainer = AgentTrainer(
    config=config,
    tokenizer=tokenizer,
    role_worker_mapping=role_worker_mapping,
    resource_pool_manager=resource_pool_manager,
    processor=processor,
    reward_fn=reward_fn,
    val_reward_fn=val_reward_fn,
    train_dataset=None,  # Agent-based training doesn't use datasets
    val_dataset=None,
    collate_fn=collate_fn,
    train_sampler=None,
)

# Initialize workers (PPO + multi-turn rollout)
trainer.init_workers()

# Train with multi-turn rollouts
trainer.fit()
```

## Key Features

### 1. Agent Registration System
- Agents are registered using `@register_agent("agentName")` decorator
- Agent class is automatically resolved from config
- Supports multiple agent types through registry

### 2. Multi-Agent Rollouts
- Supports batch processing of multiple agents
- Agents are grouped for training purposes (`agent_group_num` × `agent_group_size`)
- Each agent manages its own environment and history

### 3. Flexible Reward Management
- Custom `DummyRewardManager` for reward computation
- Supports environment-based rewards through agent metrics
- Configurable reward normalization and filtering

### 4. Environment Integration
- Agent-environment interface through `get_llm_prompts()` and `get_env_outputs()`
- Support for complex environments like Sokoban
- Configurable action spaces and state representations

## Benefits

1. **Minimal Changes**: Only replaces the generation step, everything else stays the same
2. **Compatibility**: Maintains full compatibility with existing PPO infrastructure
3. **Flexibility**: Easy to switch between single-turn and multi-turn training
4. **Extensibility**: Can easily add more complex multi-turn logic without affecting PPO
5. **Agent-Centric**: Each agent manages its own environment and history internally
6. **Modular Design**: Clean separation between PPO training and agent management

## Implementation Complete ✅

The implementation is fully functional with:
- ✅ Complete `AgentTrainer` class with all modifications
- ✅ Full `SyncMultiTurnRollout` implementation
- ✅ Example `SokobanAgent` with environment integration
- ✅ Updated training script with proper imports and configurations
- ✅ Agent registration system for extensibility
- ✅ Multi-turn validation support
- ✅ Rollout filtering and reward normalization

## Key Insight

The core insight is that we only need to replace **one function call** in the entire PPO pipeline:
- `self.actor_rollout_wg.generate_sequences()` → `self._generate_multi_turn_sequences()`
- Everything else (rewards, advantages, policy updates) remains exactly the same
- This makes the modification both simple and robust 