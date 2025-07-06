from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict

@dataclass
class BlocksworldEnvConfig:
    num_blocks: int = 3
    max_steps: int = 20
    # grid_vocab: Optional[Dict[str, str]] = field(default_factory=lambda: {
    #     "0": "table",
    #     "1": "block 1",
    #     "2": "block 2",
    #     "3": "block 3"
    # })
    # action_lookup: Optional[Dict[int, str]] = field(default_factory=lambda: {
    #     0: "table",
    #     1: "block 1",
    #     2: "block 2",
    #     3: "block 3"
    # })
    render_mode: str = "text" # "text", "1d", "2d_compact", "2d_sparse"
