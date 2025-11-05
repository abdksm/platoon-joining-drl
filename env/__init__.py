from gymnasium.envs.registration import register
from env.platoon_env import PlatoonEnv
from .config import TRAIN_CONFIG, ENV_CONFIG

register(
    id="PlatoonEnv-v0",
    entry_point=PlatoonEnv,
    max_episode_steps=1000,
)