from typing import Optional, List, Dict
from dataclasses import dataclass, field

@dataclass
class GSM8KEnvConfig:
    """Configuration for GSM8K environment"""
    dataset_path: str = field(default="openai/gsm8k")
    split: str = field(default="train")
    max_steps: int = 10