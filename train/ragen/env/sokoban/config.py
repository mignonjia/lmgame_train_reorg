from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict

# Import shared symbols from evaluation environment core
try:
    from gamingagent.envs.custom_02_sokoban.sokobanCore import (
        ROOM_STATE_TO_CHAR, 
        GRID_VOCAB, 
        ACTION_LOOKUP
    )
    # Use shared symbols
    DEFAULT_GRID_LOOKUP = ROOM_STATE_TO_CHAR
    DEFAULT_GRID_VOCAB = GRID_VOCAB  
    DEFAULT_ACTION_LOOKUP = ACTION_LOOKUP
except ImportError:
    # Fallback to local definitions if core not available
    DEFAULT_GRID_LOOKUP = {
        0: "#",  # Wall
        1: " ",  # Empty / floor
        2: "?",  # Target
        3: "*",  # Box on target
        4: "$",  # Box
        5: "@",  # Player
        6: "+",  # Player on target
    }
    DEFAULT_GRID_VOCAB = {
        "#": "wall",
        " ": "empty",
        "?": "target",
        "*": "box on target",
        "$": "box",
        "@": "player",
        "+": "player on target",
    }
    DEFAULT_ACTION_LOOKUP = {
        0: "no operation",
        1: "up", 
        2: "down", 
        3: "left", 
        4: "right"
    }

@dataclass
class SokobanEnvConfig:
    dim_room: Tuple[int, int] = (6, 6)
    max_steps: int = 100
    num_boxes: int = 3
    search_depth: int = 300
    grid_lookup: Optional[Dict[int, str]] = field(default_factory=lambda: DEFAULT_GRID_LOOKUP)
    grid_vocab: Optional[Dict[str, str]] = field(default_factory=lambda: DEFAULT_GRID_VOCAB)
    action_lookup: Optional[Dict[int, str]] = field(default_factory=lambda: DEFAULT_ACTION_LOOKUP)
    dim_x: Optional[int] = None
    dim_y: Optional[int] = None
    render_mode: str = "text"

    def __post_init__(self):
        if self.dim_x is not None and self.dim_y is not None:
            self.dim_room = (self.dim_x, self.dim_y)
            delattr(self, 'dim_x')
            delattr(self, 'dim_y')        
