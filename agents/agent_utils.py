from typing import Tuple, List, Dict, Any, Optional
import re
from dataclasses import dataclass, field
import random
import numpy as np
from contextlib import contextmanager

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

def parse_model_response(llm_response: str, enable_think: bool = True, action_sep: str = "||") -> Tuple[str, list]:
    """
    Parse model response into processed llm_response and actions.
    
    Args:
        llm_response: Raw LLM response string
        enable_think: Whether to expect <think> tags
        action_sep: Separator between actions
        
    Returns:
        Tuple[str, list]: (processed_llm_response, actions)
            - processed_llm_response: Cleaned response with proper formatting
            - actions: List of extracted action strings
    """
    pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>' if enable_think else r'<answer>(.*?)</answer>'
    match = re.search(pattern, llm_response, re.DOTALL)

    if not match:
        # No proper formatting found, return original response and empty actions
        return llm_response, []
    else:
        if enable_think:
            thought, action_content = match.group(1), match.group(2)
        else:
            thought, action_content = "", match.group(1)

    # Clean up special tokens from content
    for special_token in ["<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>"]:
        action_content = action_content.replace(special_token, "").strip()
        thought = thought.replace(special_token, "").strip()
    
    # Extract actions from cleaned content
    actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
    
    # Reconstruct properly formatted response
    if enable_think:
        processed_response = f"<think>{thought}</think><answer>{action_content}</answer>"
    else:
        processed_response = f"<answer>{action_content}</answer>"
    
    return processed_response, actions