"""
This is the context manager for the LLM agent.
author: Kangrui Wang, Zihan Wang
date: 2025-03-30
"""
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import re
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from transformers import AutoTokenizer, AutoProcessor
import hydra
from lmgame.utils import register_resolvers
from lmgame.env import REGISTERED_ENV_CONFIGS
from tensordict import TensorDict

from dataclasses import asdict
register_resolvers()

def get_masks_and_scores(input_ids: torch.Tensor, tokenizer: AutoTokenizer, all_scores: List[List[float]] = None, use_turn_scores: bool = False):
    """
    input_ids: shape (bsz, seq_len)
    Get loss mask that only learns between <|im_start|>assistant and <|im_end|>. Currently only supports qwen.
    NOTE: important! This assumes that the input_ids starts with system and then user & assistant in alternative ways
    """
    special_token = tokenizer.encode("<|im_start|>")[0]
    turn_starts = torch.where(input_ids == special_token, 1, 0)
    turn_indicators = torch.cumsum(turn_starts, dim=-1)
    response_mask = (turn_indicators % 2 == 1) & (turn_indicators > 1) # only learns all assistant turns
    loss_mask = (turn_indicators > 1) # learns everything after system prompt

    reward_token = tokenizer.encode("<|im_end|>")[0]
    score_tensor = torch.zeros_like(input_ids, dtype=torch.float32)
    if use_turn_scores:
        for idx, scores in enumerate(list(zip(*all_scores))):
            scores = torch.tensor(scores, dtype=torch.float32)
            turn_indicator = idx * 2 + 3 # 0: pad. 1: system. 2+2n: user. 3+2n: assistant
            reward_position = (input_ids == reward_token) & (turn_indicators == turn_indicator)
            score_tensor[reward_position] = scores
    else:
        scores = [sum(i) for i in all_scores]
        score_tensor[:, -1] = torch.tensor(scores, dtype=torch.float32)
    loss_mask = loss_mask[:, :-1] # remove the last token
    score_tensor = score_tensor[:, 1:] # remove the first token

    return loss_mask, score_tensor, response_mask



class ContextManager:
    """
    Manages the context for LLM interactions with environments.
    Translates between environment outputs and LLM inputs, and vice versa.
    """

    def __init__(self, 
                 config,
                 tokenizer,
                 processor = None,
                 mode: str = "train",
                 ):
        """
        Initialize the ContextManager.
        Processor is used to process the image data.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.action_sep = self.config.agent_proxy.action_sep
        self.special_token_list = ["<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>"]

        self.es_cfg = self.config.es_manager[mode]
        self.env_nums = {
                env_tag: n_group * self.es_cfg.group_size
                for n_group, env_tag in zip(self.es_cfg.env_configs.n_groups, self.es_cfg.env_configs.tags)
        }
        self._init_prefix_lookup()
    
    def _init_prefix_lookup(self):
        prefix_lookup = {}
        prefixes = {}
        env_config_lookup = {}
        env_config = {}
        for env_tag, env_config in self.config.custom_envs.items():
            if env_tag not in self.es_cfg.env_configs.tags:
                continue
            env_config_new = asdict(REGISTERED_ENV_CONFIGS[env_config.env_type]())
            for k,v in env_config.items():
                env_config_new[k] = v
            env_instruction = env_config_new.get("env_instruction", "")
            
            # Skip detailed symbol descriptions if processor is available (image mode)
            if env_config_new.get("grid_vocab", False) and self.processor is None:
                grid_vocab_str = "\nThe meaning of each symbol in the state is:\n" + ", ".join([f"{k}: {v}" for k, v in env_config_new["grid_vocab"].items()])
                env_instruction += grid_vocab_str
            elif self.processor is not None:
                # For image mode, provide concise visual analysis instructions
                env_instruction += "\nYou can see the game state visually in the provided image. First carefully examine the layout to identify your position, box locations, target positions, and walls. Then plan your moves by reasoning about how each action will change the board state and bring boxes closer to targets."
                
            if env_config_new.get("action_lookup", False):
                action_lookup_str = "\nYour available actions are:\n" + ", ".join([f"{v}" for k, v in env_config_new["action_lookup"].items()])
                action_lookup_str += f"\nYou can make up to {env_config_new.get('max_actions_per_traj', 5)} actions, separated by the action separator \" " + self.action_sep + " \"\n"
                env_instruction += action_lookup_str
            prefixes[env_tag] = env_instruction
            env_config_lookup[env_tag] = {'max_tokens': env_config.get("max_tokens", self.config.actor_rollout_ref.rollout.response_length)}

        tags = self.es_cfg.env_configs.tags
        n_groups = self.es_cfg.env_configs.n_groups
        group_size = self.es_cfg.group_size

        cur_group = 0
        for env_tag, n_group in zip(tags, n_groups):
            env_instruction = prefixes[env_tag]
            start_idx = cur_group * group_size
            end_idx = (cur_group + n_group) * group_size
            for i in range(start_idx, end_idx):
                prefix_lookup[i] = env_instruction
                env_config_lookup[i] = env_config_lookup[env_tag]
                env_config_lookup[i]['tag'] = env_tag
            cur_group += n_group
            
        self.prefix_lookup = prefix_lookup
        self.env_config_lookup = env_config_lookup

    def _parse_response(self, response: str, env_tag: str) -> List:
        if "NoThink" in env_tag:
            return response, [response]
        pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>' if self.config.agent_proxy.enable_think else r'<answer>(.*?)</answer>'
        match = re.search(pattern, response, re.DOTALL)
        if not match:
            # think_content, action_content, actions = "", "", [] # do not remove this kind of invalid string
            llm_response, actions = response, []
        else:
            if self.config.agent_proxy.enable_think:
                think_content, action_content = match.group(1), match.group(2)
            else:
                think_content, action_content = "", match.group(1)

                
            for special_token in self.special_token_list:
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()
            
            actions = [action.strip() for action in action_content.split(self.action_sep) if action.strip()]
            max_actions = self.config.agent_proxy.max_actions_per_turn

            if len(actions) > max_actions:
                actions = actions[:max_actions] #Only the first MAX_ACTIONS actions are kept in the rollout.
                action_content = (" " + self.action_sep + " ").join(actions)

            llm_response = f"<think>{think_content}</think><answer>{action_content}</answer>" if self.config.agent_proxy.enable_think else f"<answer>{action_content}</answer>"
        return llm_response, actions
        
    def _normalize_score_tensor(self, score_tensor: torch.Tensor, env_outputs: List[Dict]) -> torch.Tensor:
        """
        Normalize the score tensor to be between 0 and 1.
        NOTE: only support score at the last token for now
        """
        assert self.config.agent_proxy.use_turn_scores == False, "Reward normalization is not supported for use_turn_scores == True"
        
        rn_cfg = self.config.agent_proxy.reward_normalization
        grouping, method = rn_cfg.grouping, rn_cfg.method
        if grouping == "state":
            group_tags = [env_output["group_id"] for env_output in env_outputs]
        elif grouping == "inductive":
            group_tags = [env_output["tag"] for env_output in env_outputs]
        elif grouping == "batch":
            group_tags = [1] * len(env_outputs)
        else:
            raise ValueError(f"Invalid grouping: {grouping}")


        if method == "mean_std":
            norm_func = lambda x: (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6) if x.std(dim=-1, keepdim=True).abs().max() > 1e-6 else torch.zeros_like(x) # stable to bf16 than x.std()
        elif method == "mean":
            norm_func = lambda x: (x - x.mean(dim=-1, keepdim=True))
        elif method == "asym_clip":
            norm_func = lambda x: ((x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6) if x.std(dim=-1, keepdim=True).abs().max() > 1e-6 else torch.zeros_like(x)).clamp(min=-1, max=3)
        elif method == "identity":
            norm_func = lambda x: x
        else:
            raise ValueError(f"Invalid normalization method: {method}")

        # apply groupwise normalization
        group2index = {}
        for i, env_tag in enumerate(group_tags):
            if env_tag not in group2index:
                group2index[env_tag] = []
            group2index[env_tag].append(i)
        group2index = {k: torch.tensor(v) for k, v in group2index.items()}

        
        acc_scores = score_tensor[:, -1]
        normalized_acc_scores = acc_scores.clone()
        for group, index in group2index.items():
            normalized_acc_scores[index] = norm_func(normalized_acc_scores[index])

        # apply penalty
        penalty = torch.tensor([env_output["penalty"] for env_output in env_outputs], dtype=torch.float32)
        normalized_acc_scores = normalized_acc_scores + penalty

        score_tensor[:, -1] = normalized_acc_scores

        return score_tensor
    
    def get_lm_inputs(self, env_outputs: List[Dict], prepare_for_update: bool) -> DataProto:
        """
        env_outputs - please see below example
        [
            {"env_id": 1, "history": [{"state": "###\n#x_#", "llm_response": "Response 1", "reward": 0.5}, {"state": "###\n#x_#"}]},
            {"env_id": 2, "history": [{"state": "###\n#x_#"}]},
            ...
        ]
        prefix_lookup - from env_id to initial prompt
        """
        llm_input_texts = []
        messages_list = [] # for api calling
        all_multi_modal_data = []  # Store image data for each env_output
        all_multi_modal_inputs = []  # Store processed multi-modal inputs
        debug_raw_prompts = []  # Store final prompt text (after vision token replacement) for debugging
        
        for env_output in env_outputs:
            env_id = env_output['env_id']
            env_tag = self.env_config_lookup[env_id]['tag']
            if 'state' in env_output['history'][-1] and prepare_for_update:
                env_output['history'] = env_output['history'][:-1] # when prepare for update, we do not add the state from the n+1 turn to the trajectory
            messages = [
                {"role": "system", "content": f"You're a helpful assistant. "}, 
                {"role": "user", "content": self.prefix_lookup[env_output["env_id"]]}
            ]

            # Collect images from the state if any
            multi_modal_data = {"image": [], "video": []}

            for idx, content in enumerate(env_output["history"]):
                if "NoThink" not in env_tag and "NoAction" not in env_tag:
                    messages[-1]["content"] += f"\nTurn {idx + 1}:\n"
                if "state" in content:
                    FORMAT_PROMPT = "<think> [Your thoughts] </think> <answer> [your answer] </answer>" if self.config.agent_proxy.enable_think else "<answer> [your answer] </answer>"
                    LENGTH_PROMPT = f"Max response length: {self.env_config_lookup[env_output['env_id']]['max_tokens']} words (tokens)."
                    
                    state_content = content['state']
                    
                    # Check if state contains image data (RGB array, PIL images, or <image> token)
                    if self.processor is not None:
                        # Check for images from es_manager (PIL Images list)
                        if 'images' in content and content['images'] is not None:
                            # Images from es_manager - already PIL Images
                            num_images = len(content['images'])
                            print(f"🖼️  ContextManager: Found {num_images} images in env {env_output['env_id']} turn {idx}")
                            for pil_image in content['images']:
                                multi_modal_data["image"].append(pil_image)
                            # For Qwen2.5-VL, use proper image token format
                            # Remove existing tokens and add CORRECT NUMBER of <image> tokens
                            state_text = state_content.replace('<image>', '').replace('<images>', '').strip()
                            # Add exactly the right number of <image> tokens for VLLM
                            image_tokens = "<image> " * num_images
                            state_text = image_tokens.strip() + (" " + state_text if state_text else "")
                            print(f"🔧 ContextManager: Generated {num_images} <image> tokens: '{image_tokens.strip()}'")
                        # Check for direct RGB image array in content (backward compatibility)
                        elif 'image' in content and content['image'] is not None:
                            # Image passed as RGB array or PIL Image
                            image_data = content['image']
                            if hasattr(image_data, 'shape'):  # numpy array
                                from PIL import Image
                                # Convert RGB array to PIL Image
                                if len(image_data.shape) == 3 and image_data.shape[2] == 3:
                                    image = Image.fromarray(image_data.astype('uint8'))
                                else:
                                    image = Image.fromarray(image_data.astype('uint8'))
                            else:
                                # Assume it's already a PIL Image
                                image = image_data
                            
                            multi_modal_data["image"].append(image)
                            # For Qwen2.5-VL, use proper image token format
                            state_text = state_content.replace('<image>', '').strip() if '<image>' in state_content else state_content
                            # Add exactly one <image> token for single image
                            state_text = "<image>" + (" " + state_text if state_text else "")
                        elif '<image>' in state_content or '<images>' in state_content:
                            # State contains image token but no actual image data
                            state_text = state_content
                        else:
                            state_text = state_content
                    else:
                        state_text = state_content
                    
                    if "NoAction" in env_tag:
                        messages[-1]["content"] += f"Question:\n{state_text}\nAlways output: {FORMAT_PROMPT} with no extra text. Strictly follow this format.\n"
                    elif "NoThink" in env_tag:
                        messages[-1]["content"] += f"Question:\n{state_text}\n"
                    else:
                        messages[-1]["content"] += f"State:\n{state_text}\nYou have {content['actions_left']} actions left. Always output: {FORMAT_PROMPT} with no extra text. Strictly follow this format. {LENGTH_PROMPT}\n"
                    
                    # Debug log to see what's being added to messages
                    if self.processor is not None and '<image>' in state_text:
                        print(f"🔍 ContextManager: Added state with <image> token: '{state_text}'")
                if "llm_response" in content:
                    messages.append({"role": "assistant", "content": content["llm_response"]})
                if "reward" in content and not (prepare_for_update and idx == len(env_output["history"]) - 1):
                    # when prepare for update, we do not add the reward from the n+1 turn to the trajectory
                    messages.append({"role": "user", "content": f"Reward:\n{content['reward']}\n"})
                    

            # NOTE: this assertion is important for loss mask computation        
            assert all(msg["role"] == "assistant" for msg in messages[2::2])

            # Process with processor if available and images are present
            if self.processor is not None and len(multi_modal_data["image"]) > 0:
                # Use processor for multimodal input
                raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=(not prepare_for_update), tokenize=False)
                # Replace <image> placeholder with vision tokens expected by Qwen-VL
                vision_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
                raw_prompt = raw_prompt.replace("<image>", vision_placeholder)
                print(f"🔍 ContextManager: Generated raw_prompt with {len(multi_modal_data['image'])} images (vision placeholders inserted)")
                print(f"🔍 Vision placeholders count: {raw_prompt.count(vision_placeholder)}")
                
                if not prepare_for_update:
                    if "NoThink" not in env_tag:
                        if self.config.agent_proxy.enable_think:
                            raw_prompt += "<think>" # force the LLM to think before answering
                        else:
                            raw_prompt += "<answer>" # force the LLM to answer
                
                # Store prompt text for later debugging
                llm_input_texts.append(raw_prompt)
                debug_raw_prompts.append(raw_prompt)  # keep full prompt with <|vision_start|> tokens
                all_multi_modal_data.append(multi_modal_data)
                
                # Save raw prompt token ids so that we can feed vLLM without losing placeholders
                raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
                all_multi_modal_inputs.append({"raw_prompt_ids": raw_prompt_ids})
                
                # Process multimodal inputs
                images = multi_modal_data["image"] if multi_modal_data["image"] else None
                videos = multi_modal_data["video"] if multi_modal_data["video"] else None
                
                model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")
                
                # Remove input_ids and attention_mask as they'll be processed separately
                multi_modal_inputs = {k: v for k, v in model_inputs.items() if k not in ["input_ids", "attention_mask"]}
                all_multi_modal_inputs.append(multi_modal_inputs)
            else:
                # Use tokenizer for text-only input
            text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=(not prepare_for_update), tokenize=False)
            if not prepare_for_update:
                if "NoThink" not in env_tag:
                    if self.config.agent_proxy.enable_think:
                        text += "<think>" # force the LLM to think before answering
                    else:
                        text += "<answer>" # force the LLM to answer
            llm_input_texts.append(text)
                debug_raw_prompts.append(text)  # text-only prompt
                all_multi_modal_data.append({})
                all_multi_modal_inputs.append({})
            
            messages_list.append(messages)

        # Tokenize all texts together
        inputs = self.tokenizer(llm_input_texts, return_tensors="pt", padding=True, padding_side="left", truncation=False) # do not truncate here. Process later at TODO
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        position_ids = attention_mask.cumsum(dim=-1)
        
        if prepare_for_update:
            scores = [[i['reward'] for i in env_output['history']] for env_output in env_outputs]
            loss_mask, score_tensor, response_mask = get_masks_and_scores(input_ids, self.tokenizer, scores, use_turn_scores=self.config.agent_proxy.use_turn_scores)
            normalized_score_tensor = self._normalize_score_tensor(score_tensor, env_outputs)
            response_length = response_mask.sum(dim=-1).float().mean().item()

        llm_inputs = DataProto()
        llm_inputs.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": input_ids[:, 1:], # remove the first token
        }, batch_size=input_ids.shape[0])

        if prepare_for_update:
            llm_inputs.batch["loss_mask"] = loss_mask # remove the first token
            llm_inputs.batch["rm_scores"] = normalized_score_tensor # remove the first token

        llm_inputs.non_tensor_batch = {
            "env_ids": np.array([env_output["env_id"] for env_output in env_outputs], dtype=object),
            "group_ids": np.array([env_output["group_id"] for env_output in env_outputs], dtype=object),
            "messages_list": np.array(messages_list, dtype=object),
            "debug_raw_prompts": np.array(debug_raw_prompts, dtype=object),
        }
        
        # Add multimodal data if any environment has images
        has_any_images = any(len(mmd.get("image", [])) > 0 for mmd in all_multi_modal_data)
        if has_any_images:
            total_images = sum(len(mmd.get("image", [])) for mmd in all_multi_modal_data)
            print(f"🖼️  ContextManager: Found {total_images} images across {len(env_outputs)} environments")
            print(f"🔍 ContextManager: Image counts per env: {[len(mmd.get('image', [])) for mmd in all_multi_modal_data]}")
            llm_inputs.non_tensor_batch["multi_modal_data"] = np.array(all_multi_modal_data, dtype=object)
            llm_inputs.non_tensor_batch["multi_modal_inputs"] = np.array(all_multi_modal_inputs, dtype=object)
        else:
            print(f"📝 ContextManager: No images found across {len(env_outputs)} environments")

        if prepare_for_update:
            metrics = {}
            for env_output in env_outputs:
                for key, value in env_output["metrics"].items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
            metrics = {
                key: np.sum(value) / self.env_nums[key.split("/")[0]]
                for key, value in metrics.items()
            }
            metrics["response_length"] = response_length
            llm_inputs.meta_info = {"metrics": metrics}
        return llm_inputs

    def get_env_inputs(self, lm_outputs: DataProto) -> List[Dict]:
        if lm_outputs.batch is not None and 'responses' in lm_outputs.batch.keys():
            responses = self.tokenizer.batch_decode(
                lm_outputs.batch['responses'], 
                skip_special_tokens=True
            )
        else: # dataproto has textual responses
            responses = lm_outputs.non_tensor_batch['response_texts']
            
        env_ids = lm_outputs.non_tensor_batch['env_ids']
        for idx, env_id in enumerate(env_ids):
            tag = self.env_config_lookup[env_id]['tag']
            if "NoThink" not in tag:
                responses[idx] = "<think>" + responses[idx] if self.config.agent_proxy.enable_think else "<answer>" + responses[idx] # The LLM generation does not include <think> tags. Add them back here.

        env_inputs = []
        for env_id, response in zip(env_ids, responses):
            env_tag = self.env_config_lookup[env_id]['tag']
            llm_response, actions = self._parse_response(response, env_tag)
            env_inputs.append({
                "env_id": env_id,
                "llm_raw_response": response,
                "llm_response": llm_response,
                "actions": actions,
            })
        return env_inputs

    def formulate_rollouts(self, env_outputs: List[Dict]) -> DataProto:
        llm_inputs = self.get_lm_inputs(env_outputs, prepare_for_update=True)
        return llm_inputs

    



@hydra.main(version_base = None, config_path = "../../config", config_name = "base")
def main(config):
    import json
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)

    ctx_manager = ContextManager(config=config, tokenizer=tokenizer)
    print(f"============= ctx_manager prefix =============\n {ctx_manager.prefix_lookup}")
    batch_list = [
        {
            "env_ids": 0,
            "chat_response": "<think><think></answer> 123. </think><answer> <answer> say | hi </answer></answer>",
        },
        {
            "env_ids": 1,
            "chat_response": "<think> 456. </think><answer> love ; you </answer><think> mlll nb </think><answer> lxxx ; you </answer>",
        }
    ]
    ctx_manager.action_sep_lookup = {
        0: "|",
        1: ";"
    }
    for item in batch_list:
        item["responses"] = tokenizer.encode(item["chat_response"], return_tensors="pt",max_length=512, truncation=True,padding="max_length")[0]
    batch_dict = collate_fn(batch_list)
    batch = DataProto.from_single_dict(batch_dict)
    
    # Add response_texts to non_tensor_batch for get_env_inputs to work properly
    batch.non_tensor_batch["response_texts"] = np.array([item["chat_response"] for item in batch_list], dtype=object)
    
    env_inputs = ctx_manager.get_env_inputs(batch)
    print(f"============= env_inputs =============\n {env_inputs}")
    


    env_outputs = [
        {
            "env_id": 1,
            "history": [
                {"state": "###\n#x_#<image>", "actions_left": 5, "llm_response": "Response 1", "reward": 0.5},
                {"state": "###\n#x_#<image>", "actions_left": 4, "llm_response": "Response 2", "reward": 0.8},
                {"state": "###\n#x_#<image>", "actions_left": 3}
            ],
            "group_id": 0,
            "metrics": {},
            "penalty": 0.0
        },
        {
            "env_id": 2,
            "history": [
                {"state": "###\n#x_#<image>", "actions_left": 5, "llm_response": "Response 3", "reward": 0.3},
                {"state": "###\n#x_#<image>", "actions_left": 4}
            ],
            "group_id": 1,
            "metrics": {},
            "penalty": 0.0
        }
    ]
    
    prefix_lookup = {1: "Initial prompt", 2: "Initial prompt 2"}
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    env_prompt = ctx_manager.get_lm_inputs(env_outputs, prepare_for_update=False)
    print(env_prompt)
    formulate_rollouts_rst= ctx_manager.formulate_rollouts(env_outputs)
    print(formulate_rollouts_rst)

if __name__ == "__main__":
    main()
    