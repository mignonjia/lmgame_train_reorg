from typing import Tuple
import re
from dataclasses import dataclass

@dataclass
class Trajectory:
    

def parse_model_response(llm_response: str, enable_think: bool = True, action_sep: str = "||") -> Tuple[str, str]:
    """
    Parse model response into thought and actions.
    mingjia add: maybe move to agent_utils.py
    """
    pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>' if enable_think else r'<answer>(.*?)</answer>'
    match = re.search(pattern, response, re.DOTALL)

    if not match:
        thought, actions = "", []
    else:
        if enable_think:
            thought, action_content = match.group(1), match.group(2)
        else:
            thought, action_content = "", match.group(1)

    for special_token in ["<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>"]:
        action_content = action_content.replace(special_token, "").strip()
        thought = thought.replace(special_token, "").strip()
    
    actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
    
    return thought, actions