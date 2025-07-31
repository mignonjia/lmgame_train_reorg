# Development Track & Guide

## Current Development Status

### 1. Performance Reproduction
- **Qwen 2.5-0.5B Sokoban PPO Training** ✅ **Completed** (yuxuan)
- **Qwen 2.5-7B Sokoban PPO Training** ✅ **Completed** (mingjia)

## Development Roadmap

### 2. Codebase Improvements (yuxuan, mingjia)

#### 2.1 Core Agent System
**Location:** `agents/sokobanAgent/`
- [✅] Handle ad-hoc message format fixes in `get_llm_prompts()` function
- [✅] Abstract base agent class for reusability

#### 2.2 Rollout System
**Location:** `rollout/sync_multi_turn_rollout.py`
- [✅] Debug early stop logic in multi-turn rollout
- [✅] Optimize reward computation (loss_mask, reward_mask mechanisms)
- [ ] Replace `tokenizer.encode()` with `verl_F.tokenize_and_postprocess_data()`

#### 2.3 Training System
**Location:** `trainer/agent_trainer.py`
- [✅] Add hyperparameter for validation agent number
- [✅] Debug `_validate()` function against mingjia's ragen implementation
- [✅] Checkpoint saving frequency settings
- [✅] fix is_action_valid metric issue
- [ ] integrate turn-based loss mask
- [ ] Add extra metrics and LLM generation logging to wandb

#### 2.3 Align with Paper Report Data
- [✅] Correct unstable validation curve
- [✅] Test general ability from simpleSokoban to largeSokoban 
- [ ] Integrate more envs: 
- [ ] Test general ability through all envs

#### 2.5 System Deisgn for Different Trainers
- [ ] Jax PPO trainer integration
- [ ] Abstract the framework to integrate different trainers

#### 2.6 Installation & Build
- [ ] Implement uv installation for faster package management
- [ ] Create pyproject.toml to package as `lmgame_train`

### 3. Feature Development

#### 3.1 Evaluation Integration (lanxiang, yuxuan)
- [ ] Integrate lmgame-bench evaluation
- [ ] Abstract agent classes: `env` → `baseAgent` → `gamingAgent`, `trainingAgent`

#### 3.2 Multi-Task Training (yuxuan, yixin)
- [ ] Add support for additional tasks:
  - GSM8K (math reasoning)
  - Blockworld (spatial reasoning)
  - Tetris (game environment)

#### 3.3 Advanced Features
- [ ] Vision modality support for multi-turn RL PPO training
- [ ] SFT (Supervised Fine-Tuning) trainer
- [ ] Asynchronous multi-turn rollout system

## Contributing

When working on any of these items:
1. Create a feature branch from main
2. Follow the existing code style and patterns
3. Add appropriate tests and documentation
4. Submit a pull request with clear description of changes

## Notes

- Priority should be given to completing the 7B model performance reproduction
- Codebase improvements should focus on maintainability and performance
- New features should be developed incrementally with proper testing
