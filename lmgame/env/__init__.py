from .sokoban.env import SokobanEnvActor
from .tetris.env import TetrisEnvActor
from .blocksworld.env import BlocksworldEnvActor
from .gsm8k.env import GSM8KEnvActor


REGISTERED_ENV_ACTORS = {
    'sokoban': SokobanEnvActor,
    'tetris': TetrisEnvActor,
    'blocksworld': BlocksworldEnvActor,
    'gsm8k': GSM8KEnvActor,
}

try:
    from .webshop.env import WebShopActor
    REGISTERED_ENV_ACTORS['webshop'] = WebShopActor
except ImportError:
    pass
