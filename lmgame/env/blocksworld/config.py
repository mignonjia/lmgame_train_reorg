from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict

@dataclass
class BlocksworldEnvConfig:
    num_blocks: int = 3
    max_steps: int = 20
    render_mode: str = "text" # "text", "1d", "2d_compact", "2d_sparse"
