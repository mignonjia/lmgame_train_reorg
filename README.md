# AgentTrainer: Multi-Turn PPO Training

## Overview

`AgentTrainer` inherits from `RayPPOTrainer` and replaces single-turn generation with multi-turn rollout logic using our `SyncMultiTurnRollout` class.

## Key Design Decisions

### 1. Inheritance Strategy
- **Inherit from RayPPOTrainer**: Maintains compatibility with existing VERL infrastructure
- **Minimal modifications**: Only replace the generation logic, keep everything else the same
- **Clean separation**: Multi-turn logic is encapsulated in `SyncMultiTurnRollout`

### 2. Main Modification Point

**Original (Single-turn)**:
```python
# In RayPPOTrainer.fit()
gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
```

**Modified (Multi-turn)**:
```python
# In AgentTrainer.fit()
gen_batch_output = self._generate_multi_turn_sequences(gen_batch)
```

### 3. Integration Architecture

```
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   PPO Trainer   │───▶│   AgentTrainer      │───▶│ SyncMultiTurnRollout│
│   (Standard)    │    │   (Multi-turn)      │    │   (Agent Manager)   │
└─────────────────┘    └──────────────────────┘    └─────────────────────┘
                                │                              │
                                ▼                              ▼
                       ┌──────────────────────┐    ┌─────────────────────┐
                       │ _generate_multi_turn │    │     Agent[0..N]     │
                       │    _sequences()      │    │  (Individual Agents)│
                       └──────────────────────┘    └─────────────────────┘
```

## Step-by-Step Implementation

### Step 1: Inherit from RayPPOTrainer ✅
```python
class AgentTrainer(RayPPOTrainer):
    def __init__(self, config, tokenizer, role_worker_mapping, 
                 resource_pool_manager, agent_cls, **kwargs):
        self.agent_cls = agent_cls  # Store agent class
        super().__init__(...)       # Initialize parent
```

### Step 2: Override `init_workers()` ✅
```python
def init_workers(self):
    super().init_workers()  # Initialize PPO workers
    
    # Initialize multi-turn rollout manager
    self.multi_turn_rollout = SyncMultiTurnRollout(
        actor_rollout_wg=self.actor_rollout_wg,
        cfg=self.config,
        tokenizer=self.tokenizer,
        processor=self.processor,
        agent_cls=self.agent_cls
    )
```

### Step 3: Replace Generation Logic ✅
```python
def _generate_multi_turn_sequences(self, gen_batch: DataProto) -> DataProto:
    # Set generation parameters
    gen_batch.meta_info.update({
        "eos_token_id": self.tokenizer.eos_token_id,
        "pad_token_id": self.tokenizer.pad_token_id,
        "do_sample": True,
    })
    
    # Reset for new batch
    if hasattr(self.multi_turn_rollout, 'reset'):
        self.multi_turn_rollout.reset()
    
    # Perform multi-turn rollout
    final_env_outs = self.multi_turn_rollout.rollout()
    rollout_batch = self.multi_turn_rollout.build_update_batch()
    
    # Convert to PPO-compatible format
    output_batch = DataProto(...)
    return output_batch
```

### Step 4: Override `fit()` Method ✅
- Copy the original `fit()` method from `RayPPOTrainer`
- Replace the single line: `self.actor_rollout_wg.generate_sequences(gen_batch)`
- With: `self._generate_multi_turn_sequences(gen_batch)`
- Keep all other PPO logic unchanged

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
  
algorithm:
  adv_estimator: "gae"

# Agent-specific config  
agent:
  max_turn: 5
  agent_length: 4

env_template:
  - env_config_for_agent_0
  - env_config_for_agent_1
  # ... etc
```

## Usage Example

```python
# Initialize trainer
trainer = AgentTrainer(
    config=config,
    tokenizer=tokenizer,
    role_worker_mapping=role_worker_mapping,
    resource_pool_manager=resource_pool_manager,
    agent_cls=MyAgentClass,  # Your custom agent class
    processor=processor,
    reward_fn=reward_fn,
    # ... other PPO arguments
)

# Initialize workers (PPO + multi-turn rollout)
trainer.init_workers()

# Train with multi-turn rollouts
trainer.fit()
```

## Next Steps (TODOs)

1. **Data Format Conversion**: Implement proper conversion between `gen_batch` format and agent rollout format
2. **Agent Implementation**: Create example agent class that works with `SyncMultiTurnRollout`
3. **Config Adaptation**: Adapt configuration structure to support agent-specific parameters
4. **Testing**: Test the integration with real agent implementations
5. **Error Handling**: Add robust error handling for multi-turn rollout failures

## Benefits

1. **Minimal Changes**: Only replaces the generation step, everything else stays the same
2. **Compatibility**: Maintains full compatibility with existing PPO infrastructure
3. **Flexibility**: Easy to switch between single-turn and multi-turn training
4. **Extensibility**: Can easily add more complex multi-turn logic without affecting PPO
5. **Agent-Centric**: Each agent manages its own environment and history internally

## Key Insight

The core insight is that we only need to replace **one function call** in the entire PPO pipeline:
- `self.actor_rollout_wg.generate_sequences()` → `self._generate_multi_turn_sequences()`
- Everything else (rewards, advantages, policy updates) remains exactly the same
- This makes the modification both simple and robust 