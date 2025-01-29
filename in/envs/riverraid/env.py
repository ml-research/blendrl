from typing import Sequence
import torch
from nudge.env import NudgeBaseEnv
from ocatari.core import OCAtari
from hackatari.core import HackAtari
import numpy as np
import torch as th
from ocatari.ram.riverraid import MAX_ESSENTIAL_OBJECTS
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

from utils import load_cleanrl_envs
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


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
    """
    NUDGE environment for River Raid.
    """

    name = "riverraid"
    pred2action = {
        "noop": 0,
        "fire": 1,
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
            env_name="ALE/Riverraid-v5",
            mode="ram",
            obs_mode="ori",
            modifs=[
                ("disable_enemy_planes"),
                ("random_init"),
                ("increase_fuel_consumption"),
            ],
            rewardfunc_path="in/envs/riverraid/blenderl_reward.py",
            render_mode=render_mode,
            render_oc_overlay=render_oc_overlay,
        )
        self.env._env = make_env(self.env._env)
        self.n_actions = 6
        self.n_raw_actions = 18
        self.n_objects = len(MAX_ESSENTIAL_OBJECTS)
        self.n_features = 4  # visible, x-pos, y-pos, type
        self.seed = seed

        self.obj_offsets = {}
        offset = 0
        for obj, max_count in MAX_ESSENTIAL_OBJECTS.items():
            self.obj_offsets[obj] = offset
            offset += max_count
        self.relevant_objects = set(MAX_ESSENTIAL_OBJECTS.keys())

    def reset(self):
        raw_state, _ = self.env.reset(seed=self.seed)
        state = self.env.objects
        self.ocatari_state = state
        logic_state, neural_state = self.extract_logic_state(
            state
        ), self.extract_neural_state(raw_state)
        logic_state = logic_state.unsqueeze(0)
        return logic_state, neural_state

    def step(self, action, is_mapped=False):
        raw_state, reward, truncations, done, infos = self.env.step(action)
        state = self.env.objects
        self.ocatari_state = state
        logic_state, neural_state = self.convert_state(state, raw_state)
        logic_state = logic_state.unsqueeze(0)
        return (logic_state, neural_state), reward, done, truncations, infos

    def extract_logic_state(self, input_state):
        state = th.zeros((self.n_objects, self.n_features + 1), dtype=th.int32)
        self.bboxes = th.zeros((self.n_objects, 4), dtype=th.int32)
        obj_count = {k: 0 for k in MAX_ESSENTIAL_OBJECTS.keys()}

        for obj in input_state:
            if obj.category not in self.relevant_objects:
                continue
            idx = self.obj_offsets[obj.category] + obj_count[obj.category]
            state[idx] = th.tensor([1, *obj.center, obj.type])
            obj_count[obj.category] += 1
            self.bboxes[idx] = th.tensor(obj.xywh)
        return state

    def extract_neural_state(self, raw_input_state):
        return torch.Tensor(raw_input_state).unsqueeze(0)

    def close(self):
        self.env.close()
