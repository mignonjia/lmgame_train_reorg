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

def parse_model_response(llm_response: str, enable_think: bool = True) -> Tuple[str, List[str]]:
    """
    Parse model response into processed llm_response and action list.
    More robust parsing that handles various edge cases and malformed responses.
    
    Args:
        llm_response: Raw LLM response string
        enable_think: Whether to expect <think> tags
        
    Returns:
        Tuple[str, List[str]]: (processed_llm_response, actions_list)
            - processed_llm_response: Cleaned response with proper formatting
            - actions_list: List of parsed action strings
    """
    # First try the primary pattern
    if enable_think:
        primary_pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>'
        match = re.search(primary_pattern, llm_response, re.DOTALL)
        
        if match:
            thought, action_content = match.group(1), match.group(2)
        else:
            # Try fallback patterns for malformed responses
            think_match = re.search(r'<think>(.*?)</think>', llm_response, re.DOTALL)
            answer_match = re.search(r'<answer>(.*?)</answer>', llm_response, re.DOTALL)
            
            if think_match and answer_match:
                thought, action_content = think_match.group(1), answer_match.group(1)
            elif answer_match:  # At least we have answer tags
                thought, action_content = "", answer_match.group(1)
            else:
                # Try to extract actions from various formats
                action_content = _extract_actions_fallback(llm_response)
                thought = ""
                if not action_content:
                    return llm_response, []
    else:
        # For non-thinking mode, just look for answer tags
        answer_match = re.search(r'<answer>(.*?)</answer>', llm_response, re.DOTALL)
        
        if answer_match:
            thought, action_content = "", answer_match.group(1)
        else:
            # Try to extract actions from various formats
            action_content = _extract_actions_fallback(llm_response)
            thought = ""
            if not action_content:
                return llm_response, []

    # Clean up special tokens from content
    for special_token in ["<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>", "</thought>"]:
        action_content = action_content.replace(special_token, "").strip()
        thought = thought.replace(special_token, "").strip()
    
    # Parse actions from content using || separator
    actions = [action.strip() for action in action_content.split('||') if action.strip()]
    
    # If no valid actions found, try alternative separators
    if not actions and action_content:
        # Try comma separator
        actions = [action.strip() for action in action_content.split(',') if action.strip()]
        # Try space separator for single actions
        if not actions:
            actions = [action_content.strip()] if action_content.strip() else []
    
    # Reconstruct properly formatted response
    if enable_think:
        processed_response = f"<think>{thought}</think><answer>{action_content}</answer>"
    else:
        processed_response = f"<answer>{action_content}</answer>"

    return processed_response, actions


def _extract_actions_fallback(text: str) -> str:
    """
    Fallback function to extract actions from malformed responses.
    Looks for common action words and patterns.
    """
    # Common Sokoban actions
    action_words = ['Up', 'Down', 'Left', 'Right']
    
    # Try to find sequences of action words
    found_actions = []
    for word in action_words:
        # Case insensitive search
        if word.lower() in text.lower():
            found_actions.append(word)
    
    # Look for patterns like "Right || Left" or "Right, Left" anywhere in text
    action_pattern = r'\b(?:Up|Down|Left|Right)(?:\s*(?:\|\||,|and|then)\s*(?:Up|Down|Left|Right))*\b'
    pattern_match = re.search(action_pattern, text, re.IGNORECASE)
    
    if pattern_match:
        return pattern_match.group(0)
    elif found_actions:
        # Return first found action as fallback
        return found_actions[0]
    
    return ""