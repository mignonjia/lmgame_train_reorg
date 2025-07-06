## Environment Setup
We recommend using Conda to manage dependencies. To create the `lmgame` environment and install all required packages, including `verl` and `vllm`, run:
```bash
bash scripts/setup.sh
```

## Training Models
Default configurations are provided in:
* `config/base.yaml`
* `config/ppo_trainer.yaml` (a symbolic link to `verl/verl/trainer/config/ppo_trainer.yaml`)

To train on Sokoban, use:
```bash 
bash train_sokoban.sh
```
The model is `Qwen2.5-0.5B-Instruct` by default, which can be trained on a single GPU. The checkpoints will be stored in `checkpoints` folder.

## Evaluation
To evaluate a trained model on both in-domain and out-of-domain tasks, run:
```bash
bash eval.sh
```

Make sure to keep `trainer.project_name` and `--config-name` consistent with the training setup.

## Acknowledgement
This project is built on the RAGEN framework. Their LICENSE is included in this repository.

