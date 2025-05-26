from typing import Sequence
import torch
from nudge.env import NudgeBaseEnv
from ocatari.core import OCAtari
from hackatari.core import HackAtari
import numpy as np
import torch as th
from packaging import version

# from ocatari.ram.alien import MAX_NB_OBJECTS
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

MAX_NB_OBJECTS = {
    "Player": 1,
    "Alien": 12,
    "Pulsar": 1,
    "Rocket": 1,  # Possibly unused
    "Egg": 156,
}
MAX_NB_OBJECTS_HUD = dict(MAX_NB_OBJECTS, **{"Score": 1, "Life": 1})


from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

GYMNASIUM_VERSION = version.parse(gym.__version__)
print(f"Gymnasium version: {GYMNASIUM_VERSION}")

if GYMNASIUM_VERSION <= version.parse("0.30.0"):
    def make_env(env):
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.AutoResetWrapper(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        return env
else:
    def make_env(env):
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.Autoreset(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        return env


class NudgeEnv(NudgeBaseEnv):
    name = "alien"
    pred2action = {
        "noop": 0,
        "fire": 1,  # Flamethrower
        "up": 2,
        "right": 3,
        "left": 4,
        "down": 5,
    }
    pred_names: Sequence

    def __init__(
        self, mode: str, render_mode="rgb_array", render_oc_overlay=False, seed=None
    ):
        super().__init__(mode)
        self.env = HackAtari(
            env_name="ALE/Alien",
            mode="ram",
            obs_mode="ori",
            modifs=[],
            rewardfunc_path="in/envs/alien/blenderl_reward.py",
            render_mode=render_mode,
            render_oc_overlay=render_oc_overlay,
        )
        self.env._env = make_env(self.env._env)
        self.n_actions = 6
        self.n_raw_actions = 18
        self.n_objects = 171
        self.n_features = 4
        self.seed = seed

        self.obj_offsets = {}
        offset = 0
        for obj, max_count in MAX_NB_OBJECTS.items():
            self.obj_offsets[obj] = offset
            offset += max_count
        self.relevant_objects = set(MAX_NB_OBJECTS.keys())

    def reset(self):
        raw_state, _ = self.env.reset(seed=self.seed)
        state = self.env.objects
        logic_state, neural_state = self.extract_logic_state(
            state
        ), self.extract_neural_state(raw_state)
        return logic_state.unsqueeze(0), neural_state

    def step(self, action, is_mapped: bool = False):
        obs, reward, done, truncations, infos = self.env.step(action)
        state = self.env.objects
        raw_state = obs
        logic_state, neural_state = self.convert_state(state, raw_state)
        logic_state = logic_state.unsqueeze(0)
        return (logic_state, neural_state), reward, done, truncations, infos

    def extract_logic_state(self, input_state):
        state = th.zeros((self.n_objects, self.n_features), dtype=th.int32)

        obj_count = {k: 0 for k in MAX_NB_OBJECTS.keys()}
        for obj in input_state:
            if obj.category not in self.relevant_objects:
                continue
            idx = self.obj_offsets[obj.category] + obj_count[obj.category]
            state[idx] = th.tensor([1, *obj.center, obj.orientation or 0])
            obj_count[obj.category] += 1
        return state

    def extract_neural_state(self, raw_input_state):
        return torch.tensor(raw_input_state, dtype=torch.float32).unsqueeze(0)

    def close(self):
        self.env.close()
