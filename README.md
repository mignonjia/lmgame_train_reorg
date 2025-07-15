# LMGame Training Framework

A comprehensive framework for multi-turn reinforcement learning training of language model agents in gaming environments.

## Quick Start

### Prerequisites
- Python 3.10
- CUDA-compatible GPU (A100, L40, or similar)
- Conda package manager

### Installation

1. **Create conda environment:**
   ```bash
   conda create --name lmgame_train python=3.10
   conda activate lmgame_train
   ```

2. **Set up authentication (optional but recommended):**
   ```bash
   export WANDB_API_KEY=your_wandb_api_key
   export WANDB_ENTITY=your_wandb_entity
   export HF_TOKEN=your_huggingface_token
   ```

3. **Run setup script:**
   ```bash
   ./scripts/setup.sh
   ```

## Training Examples

### Sokoban Agent Training
```bash
source train_sokoban.sh
```

## Hardware Configuration

The framework is pre-configured for different GPU setups:

| GPU Type | Agent Groups | Group Size | Total Agents | Default Model |
|----------|--------------|------------|--------------|---------------|
| **A100** (default) | 8 | 16 | 128 | Qwen/Qwen2.5-0.5B-Instruct |
| L40 | 4 | 2 | 8 | Qwen/Qwen2.5-0.5B-Instruct |

> **Note:** The A100 configuration is the default setting in `configs/base.yaml`. For other GPUs, adjust `agent_group_num` and `agent_group_size` in the config file.

## Documentation

- **[System Design Overview](SYSTEMDESIGN.md)** - Architecture and design principles
- **[Development Guide](DEVELOPMENT.md)** - Contributing and development workflow

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.