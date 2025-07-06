import gymnasium as gym
import numpy as np
import os
import pygame # For rendering
from PIL import Image, ImageDraw, ImageFont 
from typing import Any, Dict, Tuple, Optional, List, Union
import json
import re # For parsing levels.txt

from gamingagent.envs.gym_env_adapter import GymEnvAdapter
from gamingagent.modules.core_module import Observation
from gymnasium.spaces import Discrete, Box 
from gymnasium.core import RenderFrame

# It's better to move create_board_image_sokoban to env_utils if it's generic enough,
# but for now, let's assume it's defined here or imported from a local helper.
# For this refactor, we will redefine it here, taking inspiration from sokoban_env.py.

# --- Constants and Asset Paths (from sokoban_env.py) ---
# Updated lookup: 0-4 (no-op + four directions). A single directional action will move
# the player; if a box is in the chosen direction and the square beyond is free, the
# box will be pushed automatically.
ACTION_LOOKUP = {
    0: 'Noop',
    1: 'Up',
    2: 'Down',
    3: 'Left',
    4: 'Right',
}

CHANGE_COORDINATES = { # (row_change, col_change)
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1)    # Right
}

ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets", "images")
LEVELS_FILE_PATH = os.path.join(os.path.dirname(__file__), "assets", "levels.txt")

ROOM_STATE_TO_CHAR = {
    0: '#',  # Wall
    1: '_',  # Empty / Floor
    2: 'O',  # Target
    3: '√',  # Box on target
    4: 'X',  # Box
    5: 'P',  # Player
    6: 'S',  # Player on target
}
CHAR_TO_ROOM_STATE = {v: k for k, v in ROOM_STATE_TO_CHAR.items()}


def load_sokoban_asset_image(path, size):
    if not os.path.exists(path): return None
    try:
        img = Image.open(path).convert("RGBA")
        return img.resize(size, Image.Resampling.LANCZOS)
    except Exception: return None

def create_board_image_sokoban(board_state: np.ndarray, save_path: str, tile_size: int = 32, perf_score: Optional[float] = None, action_taken_str: Optional[str] = None):
    if board_state is None:
        img = Image.new('RGB', (tile_size * 5, tile_size * 5), (128, 128, 128)) # Default small error image
        draw = ImageDraw.Draw(img)
        draw.text((10,10), "Error: No board state", fill=(255,0,0))
        if save_path:
             os.makedirs(os.path.dirname(save_path), exist_ok=True)
             img.save(save_path)
        return

    rows, cols = board_state.shape
    img_width = cols * tile_size
    img_height = rows * tile_size
    img = Image.new('RGB', (img_width, img_height), (200, 200, 200)) # Background
    draw = ImageDraw.Draw(img)

    asset_paths = {
        "wall": os.path.join(ASSET_DIR, "wall.png"),
        "floor": os.path.join(ASSET_DIR, "floor.png"),
        "box": os.path.join(ASSET_DIR, "box.png"),
        "box_on_target": os.path.join(ASSET_DIR, "box_docked.png"),
        "player": os.path.join(ASSET_DIR, "worker.png"),
        "player_on_target": os.path.join(ASSET_DIR, "worker_dock.png"),
        "target": os.path.join(ASSET_DIR, "dock.png"),
    }
    assets = {k: load_sokoban_asset_image(p, (tile_size, tile_size)) for k, p in asset_paths.items()}

    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * tile_size, r * tile_size
            tile_val = board_state[r, c] # Numerical value from room_state

            # Draw floor first
            if assets["floor"]:
                img.paste(assets["floor"], (x0, y0), assets["floor"] if assets["floor"].mode == 'RGBA' else None)
            
            asset_to_draw = None
            if tile_val == 0: asset_to_draw = assets["wall"]      # Wall
            elif tile_val == 2: asset_to_draw = assets["target"]   # Target
            elif tile_val == 3: # Box on Target
                if assets["target"]: # Draw target beneath box_on_target
                    img.paste(assets["target"], (x0, y0), assets["target"] if assets["target"].mode == 'RGBA' else None)
                asset_to_draw = assets["box_on_target"]
            elif tile_val == 4: asset_to_draw = assets["box"]      # Box
            elif tile_val == 5: asset_to_draw = assets["player"]   # Player
            elif tile_val == 6: # Player on Target
                if assets["target"]: # Draw target beneath player
                    img.paste(assets["target"], (x0, y0), assets["target"] if assets["target"].mode == 'RGBA' else None)
                asset_to_draw = assets["player_on_target"]
            
            if asset_to_draw:
                img.paste(asset_to_draw, (x0, y0), asset_to_draw if asset_to_draw.mode == 'RGBA' else None)
            elif tile_val == 1: # Floor, already drawn
                pass
            else: # Fallback for unknown tile values
                draw.rectangle([x0, y0, x0 + tile_size, y0 + tile_size], fill=(100, 100, 100))
                draw.text((x0 + 5, y0 + 5), ROOM_STATE_TO_CHAR.get(tile_val, "?"), fill=(255,255,255))

    common_font = None
    try:
        font_size = max(10, tile_size // 2 - 2)
        common_font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        common_font = ImageFont.load_default(size=font_size if 'font_size' in locals() else 12)

    if perf_score is not None:
        text_content = f"Perf: {perf_score:.2f}"
        shadow_offset = 1
        draw.text((5 + shadow_offset, 5 + shadow_offset), text_content, fill=(0,0,0), font=common_font)
        draw.text((5, 5), text_content, fill=(255, 255, 255), font=common_font)

    if action_taken_str is not None:
        action_text_content = f"Action: {action_taken_str}"
        shadow_offset = 1
        # Calculate text size to position at bottom
        if hasattr(common_font, 'getbbox'):
            bbox = common_font.getbbox(action_text_content)
            text_h = bbox[3] - bbox[1]
        elif hasattr(common_font, 'getsize'): # Legacy
            _, text_h = common_font.getsize(action_text_content)
        else: # Fallback
            text_h = font_size if 'font_size' in locals() else 12

        text_x = 5
        text_y = img_height - text_h - 5
        draw.text((text_x + shadow_offset, text_y + shadow_offset), action_text_content, fill=(0,0,0), font=common_font)
        draw.text((text_x, text_y), action_text_content, fill=(255,255,255), font=common_font)
            
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir: os.makedirs(save_dir, exist_ok=True)
        img.save(save_path)

# ------------------------- Sokoban 1989 Evaluation Environment -------------------------

class SokobanEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array', 'raw'], 'render_fps': 4}

    def __init__(self,
                 render_mode: Optional[str] = None,
                 # Sokoban specific params from game_env_config.json
                 dim_room: Tuple[int, int] = (10, 10),
                 max_steps_episode: int = 200,
                 num_boxes: int = 3,
                 num_gen_steps: Optional[int] = None, # For procedural generation
                 level_to_load: Optional[int] = None, # To load a specific level
                 tile_size_for_render: int = 32,
                 # Adapter related parameters (will be passed by the runner)
                 game_name_for_adapter: str = "sokoban",
                 observation_mode_for_adapter: str = "vision",
                 agent_cache_dir_for_adapter: str = "cache/sokoban/default_run",
                 game_specific_config_path_for_adapter: str = "gamingagent/envs/custom_02_sokoban/game_env_config.json",
                 max_stuck_steps_for_adapter: Optional[int] = 20
                 ):
        
        self.dim_room = dim_room
        self.max_steps_episode = max_steps_episode
        self.num_boxes_initial = num_boxes # Store initial config
        self.level_to_load = level_to_load
        self.tile_size_for_render = tile_size_for_render
        self.current_level = level_to_load if level_to_load is not None else 1
        self.max_level = 6  # Maximum level number in levels.txt
        
        if num_gen_steps is None and not self.level_to_load : 
            self.num_gen_steps = int(1.7 * (self.dim_room[0] + self.dim_room[1]))
        else:
            self.num_gen_steps = num_gen_steps if num_gen_steps is not None else 0 # default if level_to_load

        # Penalties and Rewards from sokoban_env_old.py
        self.penalty_for_step = -0.1
        self.penalty_box_off_target = -1.0 
        self.reward_box_on_target = 1.0
        self.reward_finished = 10.0
        
        self.action_space = Discrete(5) # 0: no_op, 1-4: directional moves (push happens automatically if possible)
        # Observation space will be dynamically set in reset based on room_fixed and tile_size
        # For now, set a placeholder. The adapter gets Observation objects.
        # The gym observation space for this env will be the raw board state (numerical)
        self.observation_space = Box(low=0, high=6, shape=self.dim_room, dtype=np.uint8) 

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None # For pygame rendering
        self.clock = None  # For pygame rendering

        # Initialize Adapter
        self.adapter = GymEnvAdapter(
            game_name=game_name_for_adapter,
            observation_mode=observation_mode_for_adapter,
            agent_cache_dir=agent_cache_dir_for_adapter,
            game_specific_config_path=game_specific_config_path_for_adapter, # Adapter loads its own action mapping
            max_steps_for_stuck=max_stuck_steps_for_adapter
        )

        # Sokoban-specific performance score tracking
        self.previous_boxes_on_target_for_perf: int = 0
        self.current_episode_cumulative_perf_score: float = 0.0

        self.room_fixed: Optional[np.ndarray] = None
        self.room_state: Optional[np.ndarray] = None
        self.player_position: Optional[np.ndarray] = None
        self.num_boxes_current: int = 0 # Number of boxes in the current level
        self.boxes_on_target: int = 0
        self.num_env_steps: int = 0
        self.current_reward_last_step: float = 0.0
        
        self.predefined_levels: Dict[int, List[str]] = {}
        self._load_predefined_levels() # Load levels from levels.txt
        
        # For gym_sokoban compatibility if used
        self.box_mapping: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None


    def _load_predefined_levels(self):
        if not os.path.exists(LEVELS_FILE_PATH): return
        with open(LEVELS_FILE_PATH, 'r') as f: content = f.read()
        
        level_blocks = re.split(r"Level\s+\d+", content)
        current_level_num = 0
        for block in level_blocks:
            if not block.strip(): continue
            current_level_num +=1
            # Corrected line splitting and stripping
            level_lines_raw = block.strip().splitlines()
            level_lines = [line.strip() for line in level_lines_raw if line.strip()]
            
            if level_lines:
                first_line_width = len(level_lines[0])
                if not all(len(line) == first_line_width for line in level_lines):
                    print(f"[SokobanEnv] Warning: Level {current_level_num} has inconsistent line widths. Skipping.")
                    continue
                self.predefined_levels[current_level_num] = level_lines

    def _parse_level_data(self, level_str_lines: List[str]) -> bool:
        rows = len(level_str_lines)
        if rows == 0: return False
        cols = len(level_str_lines[0])
        if cols == 0: return False

        self.dim_room = (rows, cols)
        _room_fixed = np.ones(self.dim_room, dtype=np.uint8) # Default to floor (1)
        _player_pos_list = []
        _num_boxes = 0

        # First pass: Set up _room_fixed (walls, targets, default floor for dynamic items)
        for r in range(rows):
            for c in range(cols):
                char = level_str_lines[r][c]
                if char == '#': # Wall
                    _room_fixed[r, c] = 0
                elif char in ['O', '?']: # Target (new 'O' or legacy '?')
                    _room_fixed[r, c] = 2
                # For 'X', 'P', 'S', the underlying _room_fixed is floor (1), which is the default.
                # If '√' implies a target, it's handled in the second pass by explicitly setting _room_fixed.

        _room_state = _room_fixed.copy() # Initialize _room_state based on fixed layout

        # Second pass: Place dynamic entities (player, boxes) onto _room_state
        for r in range(rows):
            for c in range(cols):
                char = level_str_lines[r][c]
                if char in ['X', '$']: # Box (new 'X' or legacy '$')
                    if _room_fixed[r, c] == 2: # Box placed on a target square
                        _room_state[r, c] = 3 # Box on Target
                    else: # Box on Floor
                        _room_state[r, c] = 4 # Box
                    _num_boxes += 1
                elif char in ['√', '*']: # Box explicitly on Target (new '√' or legacy '*')
                    _room_fixed[r, c] = 2 # Ensure underlying fixed map shows a target
                    _room_state[r, c] = 3 # Box on Target
                    _num_boxes += 1
                elif char in ['P', '@']: # Player on Floor (new 'P' or legacy '@')
                    _room_state[r, c] = 5  # Player on Floor
                    _player_pos_list.append(np.array([r, c]))
                elif char in ['S', '+']: # Player explicitly on Target (new 'S' or legacy '+')
                    _room_fixed[r, c] = 2
                    _room_state[r, c] = 6
                    _player_pos_list.append(np.array([r, c]))
                # Accept both new and legacy character sets for backward compatibility.
        
        self.room_fixed = _room_fixed
        self.room_state = _room_state
        self.num_boxes_current = _num_boxes

        if not _player_pos_list: # No player 'P' found in level string
            print("[SokobanEnv] Warning: No player 'P' found in level. Attempting to place player on an available floor or target square.")
            # Try to find a floor or target square to place the player
            available_squares = np.argwhere((self.room_state == 1) | (self.room_state == 2)) # Floor or Empty Target
            if available_squares.size > 0:
                # Place player randomly on one of the available squares
                self.player_position = available_squares[np.random.choice(len(available_squares))]
            else: # Absolute fallback: try center; this might be a wall.
                print("[SokobanEnv] Warning: No floor or target squares available for fallback player placement. Placing at center.")
                self.player_position = np.array([self.dim_room[0] // 2, self.dim_room[1] // 2])

            r_p, c_p = self.player_position
            # Check if chosen position is valid and not a wall before placing player state
            if self.room_fixed[r_p, c_p] != 0: # If not a wall
                 self.room_state[r_p, c_p] = 6 if self.room_fixed[r_p, c_p] == 2 else 5
            else: # Chosen fallback is a wall, this is bad. Player is effectively stuck or invalid.
                 print(f"[SokobanEnv] CRITICAL: Fallback player position ({r_p},{c_p}) is a wall. Level may be unplayable.")
                 # Keep player_position, but room_state at wall remains wall. Agent might get stuck immediately.
        elif len(_player_pos_list) > 1:
            print(f"[SokobanEnv] Warning: Multiple players ({len(_player_pos_list)}) found in level. Using the first one.")
            self.player_position = _player_pos_list[0]
        else: # Exactly one player found
            self.player_position = _player_pos_list[0]

        # Update observation space based on actual loaded level dimensions
        self.observation_space = Box(low=0, high=6, shape=self.dim_room, dtype=np.uint8)
        return True

    def _generate_procedural_level(self):
        # This uses the generate_room from gym_sokoban if available
        # Adapted from sokoban_env_old.py
        try:
            from gym_sokoban.envs.room_utils import generate_room as gr_func
            self.room_fixed, self.room_state, self.box_mapping = gr_func(
                dim=self.dim_room,
                num_steps=self.num_gen_steps if self.num_gen_steps else int(1.7 * (self.dim_room[0]+self.dim_room[1])),
                num_boxes=self.num_boxes_initial,
                second_player=False # Not supporting second player
            )
            self.player_position = np.argwhere(self.room_state == 5)[0] # Player is 5
            self.num_boxes_current = self.num_boxes_initial
            self.observation_space = Box(low=0, high=6, shape=self.dim_room, dtype=np.uint8)
            return True
        except Exception as e:
            print(f"[SokobanEnv] Failed to generate procedural level: {e}. Fallback to empty level.")
            # Fallback to a simple empty level if generation fails
            self.dim_room = (5,5) # Smaller default
            self.room_fixed = np.ones(self.dim_room, dtype=np.uint8)
            self.room_state = np.ones(self.dim_room, dtype=np.uint8)
            self.room_fixed[0,:]=0; self.room_fixed[-1,:]=0; self.room_fixed[:,0]=0; self.room_fixed[:,-1]=0 # Walls
            self.room_state = self.room_fixed.copy()
            self.player_position = np.array([self.dim_room[0]//2, self.dim_room[1]//2])
            self.room_state[self.player_position[0], self.player_position[1]] = 5
            self.num_boxes_current = 0
            self.observation_space = Box(low=0, high=6, shape=self.dim_room, dtype=np.uint8)
            return False


    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None, max_memory: Optional[int] = 10, episode_id: int = 1) -> Tuple[Observation, Dict[str, Any]]:
        super().reset(seed=seed)
        self.num_env_steps = 0
        self.current_reward_last_step = 0.0
        
        # Reset Sokoban-specific performance score trackers
        self.previous_boxes_on_target_for_perf = 0
        self.current_episode_cumulative_perf_score = 0.0

        level_loaded_ok = False
        if self.level_to_load and self.level_to_load in self.predefined_levels:
            level_data_str = self.predefined_levels[self.level_to_load]
            if self._parse_level_data(level_data_str):
                level_loaded_ok = True
        
        if not level_loaded_ok: # Fallback to procedural generation or empty
            self._generate_procedural_level()

        self.boxes_on_target = np.count_nonzero(self.room_state == 3) # Box on target is 3
        
        self.adapter.reset_episode(episode_id)
        raw_board_obs = self._get_raw_board_obs()
        info_dict = self._get_info()
        
        # Calculate initial perf score using the overridden method
        initial_perf_score = self.calculate_perf_score(0, info_dict) 
    
        img_path_for_adapter = None
        text_representation_for_adapter = None
        if self.adapter.observation_mode in ["vision", "both"]:
            img_path_for_adapter = self.adapter._create_agent_observation_path(self.adapter.current_episode_id, self.adapter.current_step_num)
            create_board_image_sokoban(raw_board_obs, img_path_for_adapter, tile_size=self.tile_size_for_render, perf_score=initial_perf_score)
        
        if self.adapter.observation_mode in ["text", "both"]:
            char_board_2d_list = [[ROOM_STATE_TO_CHAR.get(tile, '?') for tile in row] for row in raw_board_obs.tolist()]
            # text_representation_for_adapter = str(char_board_2d_list)
            text_representation_for_adapter = self.matrix_to_text_table(char_board_2d_list)

        agent_observation = self.adapter.create_agent_observation(
            img_path=img_path_for_adapter,
            text_representation=text_representation_for_adapter,
            max_memory=max_memory
        )

        if self.render_mode == "human": self._render_frame()
        return agent_observation, info_dict

    def _get_raw_board_obs(self) -> np.ndarray:
        return self.room_state.copy() if self.room_state is not None else np.zeros(self.dim_room, dtype=np.uint8)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "num_env_steps": self.num_env_steps,
            "player_position": self.player_position.tolist() if self.player_position is not None else None,
            "boxes_on_target": self.boxes_on_target,
            "num_boxes": self.num_boxes_current,
            "all_boxes_on_target": self._check_if_all_boxes_on_target(),
            "reward_last_step": self.current_reward_last_step
        }

    def _check_if_all_boxes_on_target(self) -> bool:
        if self.room_state is None or self.num_boxes_current == 0: return True # No boxes means solved
        num_boxes_off_target = np.count_nonzero(self.room_state == 4) # Count boxes not on target (state 4)
        return num_boxes_off_target == 0 and self.boxes_on_target == self.num_boxes_current
        
    def _check_if_maxsteps(self) -> bool:
        return self.num_env_steps >= self.max_steps_episode

    def _calc_reward(self) -> float:
        # Adapted from sokoban_env_old.py
        reward = self.penalty_for_step
        
        current_boxes_on_target = np.count_nonzero(self.room_state == 3) # state 3 is box_on_target
        
        if current_boxes_on_target > self.boxes_on_target:
            reward += self.reward_box_on_target * (current_boxes_on_target - self.boxes_on_target)
        elif current_boxes_on_target < self.boxes_on_target:
            # This implies a box was moved OFF a target.
            reward += self.penalty_box_off_target * (self.boxes_on_target - current_boxes_on_target)
        
        self.boxes_on_target = current_boxes_on_target
        
        if self._check_if_all_boxes_on_target():
            reward += self.reward_finished
        
        return float(reward)

    def _internal_push_or_move(self, action_idx: int) -> Tuple[bool, bool]:
        """Execute a move or push depending on what is in the chosen direction.

        The action space is:
            0: no operation
            1: up
            2: down
            3: left
            4: right

        When a directional action (1-4) is selected, the environment first checks
        the square adjacent to the player in that direction:
            • If it contains a box (3 or 4) **and** the square beyond it is free
              (1 or 2), the box is pushed and the player follows it.
            • Else, if the adjacent square is free (1 or 2), the player simply
              moves there.
            • Otherwise, the action has no effect.
        """

        moved_player, moved_box = False, False

        # No-op
        if action_idx == 0:
            return moved_player, moved_box

        # Map 1-4 → 0-3 (up, down, left, right)
        direction_idx = action_idx - 1
        change = CHANGE_COORDINATES[direction_idx]

        player_r, player_c = self.player_position
        next_r, next_c = player_r + change[0], player_c + change[1]

        # Boundary check for the player's destination
        if not (0 <= next_r < self.dim_room[0] and 0 <= next_c < self.dim_room[1]):
            return moved_player, moved_box

        next_tile = self.room_state[next_r, next_c]

        # Case 1: The adjacent square contains a box → try to push
        if next_tile in [3, 4]:
            box_dest_r, box_dest_c = next_r + change[0], next_c + change[1]

            # Boundary check for the box destination
            if not (0 <= box_dest_r < self.dim_room[0] and 0 <= box_dest_c < self.dim_room[1]):
                return moved_player, moved_box  # Can't push out of bounds

            if self.room_state[box_dest_r, box_dest_c] in [1, 2]:  # Floor or Target
                # Move the box
                self.room_state[box_dest_r, box_dest_c] = 3 if self.room_fixed[box_dest_r, box_dest_c] == 2 else 4
                moved_box = True

                # Move the player into the box's previous square
                self.room_state[next_r, next_c] = 6 if self.room_fixed[next_r, next_c] == 2 else 5
                self.room_state[player_r, player_c] = self.room_fixed[player_r, player_c]
                self.player_position = np.array([next_r, next_c])
                moved_player = True

        # Case 2: The adjacent square is empty → simple move
        elif next_tile in [1, 2]:
            self.room_state[next_r, next_c] = 6 if self.room_fixed[next_r, next_c] == 2 else 5
            self.room_state[player_r, player_c] = self.room_fixed[player_r, player_c]
            self.player_position = np.array([next_r, next_c])
            moved_player = True

        return moved_player, moved_box

    def matrix_to_text_table(self, matrix: List[List[str]]) -> str:
        """Convert a 2D list matrix into a structured text table."""
        header = "ID  | Item Type    | Position"
        line_separator = "-" * len(header)
        
        item_map = {
            '#': 'Wall',
            'P': 'Worker',
            'S': 'Worker on Dock',
            'X': 'Box',
            'O': 'Dock',
            '√': 'Box on Dock',
            '_': 'Empty'
        }
        
        table_rows = [header, line_separator]
        item_id = 1
        
        for row_idx, row in enumerate(matrix):
            for col_idx, cell in enumerate(row):
                item_type = item_map.get(cell, 'Unknown')
                table_rows.append(f"{item_id:<3} | {item_type:<12} | ({col_idx}, {row_idx})")
                item_id += 1
        
        return "\n".join(table_rows)

    def _progress_to_next_level(self) -> bool:
        """Progress to the next level if available. Returns True if progressed, False if at max level."""
        if self.current_level < self.max_level:
            self.current_level += 1
            self.level_to_load = self.current_level
            return True
        return False

    def step(self, agent_action_str: Optional[str], thought_process: str = "", time_taken_s: float = 0.0) -> Tuple[Observation, float, bool, bool, Dict[str, Any], float]:
        self.adapter.increment_step()
        
        # Map agent string action to environment action index using adapter
        env_action_idx = self.adapter.map_agent_action_to_env_action(agent_action_str)
        
        reward = 0.0
        terminated = False
        truncated = False
        
        if env_action_idx is not None and self.action_space.contains(env_action_idx):
            moved_player, moved_box = self._internal_push_or_move(env_action_idx)
            reward = self._calc_reward()
            terminated = self._check_if_all_boxes_on_target()
            
            # If level is completed, try to progress to next level
            if terminated and self._progress_to_next_level():
                # Reset the environment for the new level
                self.reset()
                # Return the new observation and info
                raw_board_obs = self._get_raw_board_obs()
                info_dict = self._get_info()
                current_perf_score = self.calculate_perf_score(reward, info_dict)
                
                img_path_for_adapter = None
                text_representation_for_adapter = None
                if self.adapter.observation_mode in ["vision", "both"]:
                    img_path_for_adapter = self.adapter._create_agent_observation_path(self.adapter.current_episode_id, self.adapter.current_step_num)
                    create_board_image_sokoban(raw_board_obs, img_path_for_adapter, tile_size=self.tile_size_for_render, perf_score=current_perf_score, action_taken_str=agent_action_str)
                
                if self.adapter.observation_mode in ["text", "both"]:
                    char_board_2d_list = [[ROOM_STATE_TO_CHAR.get(tile, '?') for tile in row] for row in raw_board_obs.tolist()]
                    text_representation_for_adapter = self.matrix_to_text_table(char_board_2d_list)

                agent_observation = self.adapter.create_agent_observation(
                    img_path=img_path_for_adapter,
                    text_representation=text_representation_for_adapter
                )
                
                return agent_observation, reward, False, False, info_dict, current_perf_score
        else: # Invalid or no action from agent
            print(f"[SokobanEnv] Action '{agent_action_str}' (mapped to {env_action_idx}) is skip/invalid. Env not stepped.")
            reward = self.penalty_for_step
            terminated = self._check_if_all_boxes_on_target()

        self.num_env_steps += 1
        truncated = self._check_if_maxsteps()
        self.current_reward_last_step = reward

        raw_board_obs = self._get_raw_board_obs()
        info_dict = self._get_info()
        current_perf_score = self.calculate_perf_score(reward, info_dict)
        
        img_path_for_adapter = None
        text_representation_for_adapter = None
        if self.adapter.observation_mode in ["vision", "both"]:
            img_path_for_adapter = self.adapter._create_agent_observation_path(self.adapter.current_episode_id, self.adapter.current_step_num)
            create_board_image_sokoban(raw_board_obs, img_path_for_adapter, tile_size=self.tile_size_for_render, perf_score=current_perf_score, action_taken_str=agent_action_str)
        
        if self.adapter.observation_mode in ["text", "both"]:
            char_board_2d_list = [[ROOM_STATE_TO_CHAR.get(tile, '?') for tile in row] for row in raw_board_obs.tolist()]
            text_representation_for_adapter = self.matrix_to_text_table(char_board_2d_list)

        agent_observation = self.adapter.create_agent_observation(
            img_path=img_path_for_adapter,
            text_representation=text_representation_for_adapter
        )
        
        final_terminated, final_truncated = self.adapter.verify_termination(agent_observation, terminated, truncated)

        self.adapter.log_step_data(
            agent_action_str=agent_action_str,
            thought_process=thought_process,
            reward=reward,
            info=info_dict,
            terminated=final_terminated,
            truncated=final_truncated,
            time_taken_s=time_taken_s,
            perf_score=current_perf_score,
            agent_observation=agent_observation
        )

        if self.render_mode == "human": self._render_frame()
        return agent_observation, reward, final_terminated, final_truncated, info_dict, current_perf_score

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]: # Type hint from gym.Env
        if self.render_mode == "rgb_array":
            return self._render_frame_rgb()
        elif self.render_mode == "human":
            self._render_frame() # Handles its own display
            return None 
        elif self.render_mode == "raw": # For debugging or specific agent needs
            return self.room_fixed, self.room_state, self.player_position, self.boxes_on_target
        return None # Default for other modes or if no rendering

    def _render_frame_rgb(self) -> Optional[np.ndarray]:
        if self.room_state is None: return None
        
        # Create a temporary path to save the image, then load it as numpy array
        # This is a bit indirect but reuses create_board_image_sokoban
        temp_img_path = os.path.join(self.adapter.agent_cache_dir, "_temp_render.png")
        create_board_image_sokoban(self.room_state, temp_img_path, self.tile_size_for_render)
        if os.path.exists(temp_img_path):
            img = Image.open(temp_img_path).convert('RGB')
            rgb_array = np.array(img)
            os.remove(temp_img_path)
            return rgb_array
        return None

    def _render_frame(self): # For human mode
        if self.room_state is None: return

        img_rgb_array = self._render_frame_rgb()
        if img_rgb_array is None: return

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((img_rgb_array.shape[1], img_rgb_array.shape[0])) # W, H
            pygame.display.set_caption(f"Sokoban - Level {self.current_level}")
        else:
            # Check if window size needs to be updated for new level
            current_size = self.window.get_size()
            new_size = (img_rgb_array.shape[1], img_rgb_array.shape[0])
            if current_size != new_size:
                self.window = pygame.display.set_mode(new_size)
            pygame.display.set_caption(f"Sokoban - Level {self.current_level}")

        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.surfarray.make_surface(img_rgb_array.swapaxes(0, 1)) # Pygame needs WxH
        self.window.blit(surf, (0, 0))
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
        self.adapter.close_log_file()
        print("[SokobanEnv] Closed.")

    def calculate_perf_score(self, reward: float, info: Dict[str, Any]) -> float:
        """
        Calculates a performance score for the current step for Sokoban.
        This is based on the cumulative count of newly placed boxes on targets 
        within the current episode.

        Args:
            reward (float): The reward received for the step (not directly used here).
            info (Dict[str, Any]): Additional information from the environment, expected
                                   to contain "boxes_on_target".

        Returns:
            float: The cumulative performance score for the episode up to this step.
        """
        current_boxes_on_target = info.get("boxes_on_target", 0)
        
        delta_boxes = 0
        if current_boxes_on_target > self.previous_boxes_on_target_for_perf:
            delta_boxes = current_boxes_on_target - self.previous_boxes_on_target_for_perf
        
        # Only add positive delta to the cumulative score
        if delta_boxes > 0:
            self.current_episode_cumulative_perf_score += float(delta_boxes)
            
        self.previous_boxes_on_target_for_perf = current_boxes_on_target
        
        return self.current_episode_cumulative_perf_score



# ------------------------- Sokoban 1989 Training Environment -------------------------

# NOTE: The previous training environment lived in `env.py`.  It is now merged here
# as `SokobanTrainEnv` so we can remove the extra file.

from gym_sokoban.envs.sokoban_env import SokobanEnv as GymSokobanEnv
from ragen.env.sokoban.config import SokobanEnvConfig
from ragen.utils import all_seed


class SokobanTrainEnv(GymSokobanEnv):
    """Light-weight training wrapper around `gym_sokoban`.

    This implementation is the same logic that used to live in
    `train/ragen/env/sokoban/env.py`, but without the `BaseDiscreteActionEnv`
    dependency.  It keeps the simple 4-direction discrete action space (1–4)
    and exposes text / RGB rendering helpers used by the RAGEN agent code.
    """

    def __init__(self, config: Optional[SokobanEnvConfig] = None, **kwargs):
        import gym  # local import to avoid mandatory dependency for evaluation

        # Configuration
        self.config = config or SokobanEnvConfig()
        self.GRID_LOOKUP = self.config.grid_lookup
        self.ACTION_LOOKUP = self.config.action_lookup
        self.search_depth = self.config.search_depth

        # Action space (1-4).  We deliberately keep the original "start=1"
        # behaviour for backward compatibility with pre-trained agents.
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(4, start=1)

        # Render mode ("text" or "rgb_array")
        self.render_mode = self.config.render_mode

        # Initialise the underlying gym-sokoban environment
        super().__init__(
            dim_room=self.config.dim_room,
            max_steps=self.config.max_steps,
            num_boxes=self.config.num_boxes,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Standard gym overrides / convenience helpers
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None):
        """Reset using the custom procedural level generator."""
        try:
            with all_seed(seed):
                from .utils import generate_room  # local import (heavy)

                (
                    self.room_fixed,
                    self.room_state,
                    self.box_mapping,
                    _,
                ) = generate_room(
                    dim=self.dim_room,
                    num_steps=self.num_gen_steps,
                    num_boxes=self.num_boxes,
                    search_depth=self.search_depth,
                )

            # House-keeping counters
            self.num_env_steps = 0
            self.reward_last = 0
            self.boxes_on_target = 0
            self.player_position = np.argwhere(self.room_state == 5)[0]

            return self.render()
        except (RuntimeError, RuntimeWarning):
            # Retry with a different seed to avoid dead procedural generators
            next_seed = abs(hash(str(seed))) % (2**32) if seed is not None else None
            return self.reset(next_seed)

    def step(self, action: int):
        """One environment step using underlying gym-sokoban."""
        previous_pos = self.player_position.copy()

        _, reward, done, _ = super().step(action)

        next_obs = self.render()
        action_effective = not np.array_equal(previous_pos, self.player_position)

        info = {
            "action_is_effective": action_effective,
            "action_is_valid": True,
            "success": self.boxes_on_target == self.num_boxes,
        }

        return next_obs, reward, done, info

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def render(self, mode: Optional[str] = None):  # type: ignore[override]
        render_mode = mode if mode is not None else self.render_mode

        if render_mode == "text":
            # Replace worker-on-target for nicer printing
            room = np.where(
                (self.room_state == 5) & (self.room_fixed == 2), 6, self.room_state
            )
            return "\n".join(
                "".join(self.GRID_LOOKUP.get(cell, "?") for cell in row)
                for row in room.tolist()
            )
        elif render_mode == "rgb_array":
            return self.get_image(mode="rgb_array", scale=1)
        else:
            raise ValueError(f"Invalid mode: {render_mode}")

    # Convenience wrappers ------------------------------------------------

    def get_all_actions(self) -> List[int]:
        return list(self.ACTION_LOOKUP.keys())

    def close(self):
        self.render_cache = None  # type: ignore[attr-defined]
        super().close()