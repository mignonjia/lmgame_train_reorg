from .sokoban.config import SokobanEnvConfig
from .sokoban.env import SokobanEnvActor
from .tetris.env import TetrisEnvActor
from .tetris.config import TetrisEnvConfig
from .blocksworld.env import BlocksworldEnvActor
from .blocksworld.config import BlocksworldEnvConfig
from .gsm8k.env import GSM8KEnvActor
from .gsm8k.config import GSM8KEnvConfig


REGISTERED_ENV_ACTORS = {
    'sokoban': SokobanEnvActor,
    'tetris': TetrisEnvActor,
    'blocksworld': BlocksworldEnvActor,
    'gsm8k': GSM8KEnvActor,
}

REGISTERED_ENV_CONFIGS = {
    'sokoban': SokobanEnvConfig,
    'tetris': TetrisEnvConfig,
    'blocksworld': BlocksworldEnvConfig,
    'gsm8k': GSM8KEnvConfig,
}

try:
    from .webshop.env import WebShopActor
    from .webshop.config import WebShopEnvConfig
    REGISTERED_ENV_ACTORS['webshop'] = WebShopActor
    REGISTERED_ENV_CONFIGS['webshop'] = WebShopEnvConfig
except ImportError:
    pass
