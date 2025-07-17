from typing import Tuple, List, Dict, Any, Optional
import re
from dataclasses import dataclass, field
import random
import numpy as np
from contextlib import contextmanager

# ─────────────────── DATA STRUCTURES ───────────────────
@dataclass
class EnvOutput:
    """Simple container for environment outputs that SyncMultiTurnRollout expects."""
    done: bool = False
    state: str = ""
    reward: float = 0.0
    info: Dict[str, Any] = None  # type: ignore
    
    def __post_init__(self):
        if self.info is None:
            self.info = {}

@dataclass
class Trajectory:
    """Simple trajectory class for storing a single step's information."""
    state: str = ""
    actions_left: int = 0
    actions: List[int] = field(default_factory=list)
    reward: float = 0.0
    info: Dict[str, Any] = field(default_factory=dict)
    llm_response: str = ""
    llm_raw_response: str = ""

@contextmanager
def all_seed(seed):
    """Context manager to set random seeds temporarily."""
    random_state = random.getstate()
    np_random_state = np.random.get_state()

    try:
        random.seed(seed)
        np.random.seed(seed)
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(np_random_state)
