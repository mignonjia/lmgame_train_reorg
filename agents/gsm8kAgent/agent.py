# ─────────────────── IMPORTS ───────────────────
import random
import yaml
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from agents.agent_utils import SingleTurnTrajectory, MultiTurnTrajectory, EnvOutput, debug_printout_in_env_output
from agents.base_agent import BaseAgent
from agents.gsm8kAgent.env import GSM8KEnv
from agents import register_agent

# ─────────────────── GSM8K AGENT ───────────────────
@register_agent("gsm8kAgent")
class GSM8KAgent(BaseAgent):
    """
    GSM8K agent that manages environment interactions and conversation history.
    Compatible with SyncMultiTurnRollout interface.
    """
    
    def __init__(self, config, group_id=0, agent_id=0, seed=None, tag=None):
        super().__init__(config, group_id, agent_id, seed, tag)
        self.initialize_env()
        if self.agent_config.get('use_custom_prompt', False):
            self.prompt = "You are solving Math problems. Let's think step by step. Always put the answer in integer at the end of your response."
            self.turn_prompt_template = """Incorrect Answer.\nQuestion:\n{state}\nPlease think again."""

    def initialize_env(self):
        """Initialize the GSM8K environment."""
        self.env = GSM8KEnv(self.env_config)

    # ─────────────────── ENV INTERACTION ───────────────────
    def get_env_outputs(self, llm_response: str):
        """
        Parse the model’s reply, send the (single) numeric answer to GSM8KEnv,
        and package the resulting EnvOutput + trajectory bookkeeping.
        """
        llm_raw_response = llm_response

        # Store raw response for debugging
        self.raw_response_list.append(llm_raw_response)
       
        self.cur_turn += 1

        processed_llm_response, actions = self.parse_llm_response(llm_raw_response, enable_think=self.enable_think)

        self.messages.append({"role": "assistant", "content": processed_llm_response})

        obs = self.env.render()
        total_reward = 0
        done = False
        info = {}  # Initialize info dictionary
        # If use_custom_prompt is True, we only use the last action as the answer
        if self.agent_config.get('use_custom_prompt', False):
            actions = [processed_llm_response]
            obs, reward, done, step_info = self.env.step(actions[-1])
            info.update(step_info)
        else:
            if len(actions) != 0:
                obs, reward, done, step_info = self.env.step(actions[-1])
                total_reward += reward
                info.update(step_info)
            else:
                self.penalty += self.format_penalty

        self.total_actions_consumed += len(actions)

        actions_left = max(0, self.max_actions_all_turns - self.total_actions_consumed)

        if self.cur_turn >= self.max_turns or self.total_actions_consumed >= self.max_actions_all_turns:
            done = True

        self.trajectory_history.add(SingleTurnTrajectory(
            state=obs,
            actions_left=actions_left,
            actions= actions,
            reward=total_reward,
            info=info,
            llm_response=processed_llm_response,
            llm_raw_response=llm_raw_response
        ))

        # debug_printout_in_env_output(self.messages, actions, self.tag)

        return EnvOutput(
            truncated=done,
            terminated=done,
            state=obs,
            reward=total_reward,
            info=info,
        )