# LMGame Multi-Turn Reinforcement Learning

A comprehensive framework for multi-turn reinforcement learning training of language model agents in gaming environments. This framework enables training language models to play complex games through reinforcement learning, developing general reasoning and decision-making capabilities.

## Quick Start

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (A100, L40, or similar)
- Conda package manager

### Installation

1. **Create conda environment:**
   ```bash
   conda create --name LMGameRL python=3.10
   conda activate LMGameRL
   ```

2. **Install submodules and package:**
   ```bash
   source scripts/install_submodules.sh
   pip install -e .
   ```

3. **Set up authentication (optional but recommended):**
   ```bash
   export WANDB_API_KEY=your_wandb_api_key
   export WANDB_ENTITY=your_wandb_entity
   export HF_TOKEN=your_huggingface_token
   ```

4. **Environment setup:**
   ```bash
   source env_setup.sh
   ```

### Optional: Install Datasets
If you want to reproduce paper results and validate BIRD SQL performance or WebShop full dataset performance:
```bash
source scripts/install_dataset.sh
```

## Quick Run

For immediate testing and quick experimentation:
```bash
source quick_train_qwen_halfb.sh
```

## Training Examples

### General Game RL Training (Paper Reported Results)

**Sokoban Agent Training:**
```bash
source examples/sokoban_ppo/qwen_7b.sh
```

**Tetris Agent Training:**
```bash
source examples/tetris_ppo/qwen_7b.sh
```

### Customizable Training Scripts

All scripts support configurable parameters:
```bash
# Basic usage with defaults
source examples/sokoban_ppo/qwen_halfb.sh

# Custom configuration
source examples/sokoban_ppo/qwen_halfb.sh "0,1" "8" "16" "64,64" "1,1" "simpleSokobanAgent" "simpleSokobanAgent,largeSokobanAgent" 2 "my_project" "my_experiment" "Qwen/Qwen2.5-0.5B-Instruct"
```

**Parameter Order:**
1. CUDA_VISIBLE_DEVICES
2. Agent Group Number
3. Agent Group Size  
4. Validation Agent Group Numbers
5. Validation Agent Group Sizes
6. Training Tasks
7. Validation Tasks
8. Number of GPUs per Node
9. Project Name
10. Experiment Name
11. Model Path

## Hardware Configuration

The framework is pre-configured for different GPU setups:

| GPU Type | Agent Groups | Group Size | Total Agents | Default Model |
|----------|--------------|------------|--------------|---------------|
| **A100** (default) | 8 | 16 | 128 | Qwen/Qwen2.5-0.5B-Instruct |
| **A100 (7B)** | 8 | 16 | 128 | Qwen/Qwen2.5-7B-Instruct |
| L40 | 4 | 8 | 32 | Qwen/Qwen2.5-0.5B-Instruct |

> **Note:** The framework automatically scales configurations based on available hardware. Adjust parameters in training scripts for optimal performance on your setup.

## Supported Games and Agents

- **Sokoban**: Puzzle-solving game requiring spatial reasoning
- **Tetris**: Real-time decision making and planning
- **GSM8K**: Mathematical reasoning tasks
- **BlocksWorld**: Logical planning and manipulation
- **WebShop**: E-commerce navigation and decision making
- **BIRD**: SQL query generation and database reasoning

## Documentation

- **[System Design Overview](docs/SYSTEMDESIGN.md)** - Architecture and design principles
- **[Development Guide](docs/DEVELOPMENT.md)** - Contributing and development workflow

## Project Structure

```
lmgame_train_reorg/
├── lmgamerl/              # Main training package
├── configs/               # Hydra configuration files
├── examples/              # Training scripts for different games
│   ├── sokoban_ppo/       # Sokoban training examples
│   └── tetris_ppo/        # Tetris training examples
├── scripts/               # Setup and utility scripts
├── docs/                  # Documentation
└── quick_train_qwen_halfb.sh  # Quick start script
```

## Acknowledgments

This framework is built upon and extends the following excellent projects:

- **[VERL](https://github.com/volcengine/verl)** - Powers the core training framework and provides the distributed training infrastructure
- **[Ragen](https://github.com/xxxx/ragen)** - Provides foundational game environment implementations and agent interfaces

We gratefully acknowledge the contributions of these projects to the open-source RL and language model training community.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.