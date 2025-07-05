from .sokoban.config import SokobanEnvConfig
from .sokoban.sokobanEnv import SokobanTrainEnv 
from .tetris.env import TetrisEnv
from .tetris.config import TetrisEnvConfig
from .blocksworld.env import BlocksworldEnv
from .blocksworld.config import BlocksworldEnvConfig
from .gsm8k.env import GSM8KEnv
from .gsm8k.config import GSM8KEnvConfig


REGISTERED_ENVS = {
    'sokoban': SokobanTrainEnv,
    'tetris': TetrisEnv,
    'blocksworld': BlocksworldEnv,
    'gsm8k': GSM8KEnv,
}

REGISTERED_ENV_CONFIGS = {
    'sokoban': SokobanEnvConfig,
    'tetris': TetrisEnvConfig,
    'blocksworld': BlocksworldEnvConfig,
    'gsm8k': GSM8KEnvConfig,
}

try:
    from .webshop.env import WebShopEnv
    from .webshop.config import WebShopEnvConfig
    REGISTERED_ENVS['webshop'] = WebShopEnv
    REGISTERED_ENV_CONFIGS['webshop'] = WebShopEnvConfig
except ImportError:
    pass
