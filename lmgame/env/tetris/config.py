from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional

@dataclass
class TetrisEnvConfig:
    dim_room: Tuple[int, int] = (4, 4)
    box_type: int = 2
    action_lookup: Optional[Dict[int, str]] = field(default_factory=lambda: {
        0: "Left",
        1: "Right",
        2: "Down"
    })
    grid_lookup: Optional[Dict[int, str]] = field(default_factory=lambda: {
        0: "_",  # empty
        1: "#",  # filled
        2: "X",  # current block
        # 3: "O"   # bottom edge
    })
    grid_vocab: Optional[Dict[str, str]] = field(default_factory=lambda: {
        "#": "filled spaces", 
        "_": "empty",
        "X": "current block",
        # "O": "bottom edge of the board"
    })
    dim_x: Optional[int] = None
    dim_y: Optional[int] = None
    render_mode: str = "text"
    max_steps: int = 100

    def __post_init__(self):
        if self.dim_x is not None and self.dim_y is not None:
            self.dim_room = (self.dim_x, self.dim_y)
            delattr(self, 'dim_x')
            delattr(self, 'dim_y')        
