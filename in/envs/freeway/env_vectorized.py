from typing import Sequence
import torch
from blendrl.env_vectorized import VectorizedNudgeBaseEnv
from ocatari.core import OCAtari
import numpy as np

import torch as th
from ocatari.ram.freeway import MAX_NB_OBJECTS
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from hackatari.core import HackAtari
from utils import load_cleanrl_envs


from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

MAX_NB_OBJECTS = {
    "Chicken": 1,  # chicken
    "Car": 10,  # up to 10 cars can be present
}


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


# python train_blenderl.py --env-name freeway --joint-training --num-steps 16 --num-envs 1 --gamma 0.99
class VectorizedNudgeEnv(VectorizedNudgeBaseEnv):
    name = "freeway"
    pred2action = {
        "noop": 0,
        "up": 1,
        "down": 2,
    }
    pred_names: Sequence

    def __init__(
        self,
        mode: str,
        n_envs: int,
        render_mode="rgb_array",
        render_oc_overlay=False,
        seed=None,
    ):
        super().__init__(mode)
        self.n_envs = n_envs
        self.envs = [
            HackAtari(
                env_name="ALE/Freeway-v5",
                mode="ram",
                obs_mode="ori",
                rewardfunc_path="in/envs/freeway/blenderl_reward.py",
                render_mode=render_mode,
                render_oc_overlay=render_oc_overlay,
            )
            for _ in range(n_envs)
        ]

        for i in range(n_envs):
            self.envs[i]._env = make_env(self.envs[i]._env)

        self.n_actions = 3
        self.n_raw_actions = 18
        self.n_objects = len(MAX_NB_OBJECTS)
        self.n_features = 6
        self.seed = seed

        self.obj_offsets = {}
        offset = 0
        for obj, max_count in MAX_NB_OBJECTS.items():
            self.obj_offsets[obj] = offset
            offset += max_count
        self.relevant_objects = set(MAX_NB_OBJECTS.keys())

    def reset(self):
        logic_states = []
        neural_states = []
        seed_i = self.seed
        for env in self.envs:
            obs, _ = env.reset(seed=seed_i)
            obs = torch.tensor(obs).float()
            state = env.objects
            logic_state, neural_state = self.extract_logic_state(
                state
            ), self.extract_neural_state(state)
            logic_states.append(logic_state)
            neural_states.append(neural_state)
            seed_i += 1
        return torch.stack(logic_states), torch.stack(neural_states)

    def step(self, actions, is_mapped: bool = False):
        observations = []
        rewards = []
        truncations = []
        dones = []
        infos = []
        logic_states = []
        neural_states = []
        for i, env in enumerate(self.envs):
            action = actions[i]
            obs, reward, truncation, done, info = env.step(action)
            obs = torch.tensor(obs).float()
            state = env.objects
            logic_state, neural_state = self.extract_logic_state(
                state
            ), self.extract_neural_state(state)
            logic_states.append(logic_state)
            neural_states.append(neural_state)
            observations.append(obs)
            rewards.append(reward)
            truncations.append(truncation)
            dones.append(done)
            infos.append(info)
        return (
            (torch.stack(logic_states), torch.stack(neural_states)),
            rewards,
            truncations,
            dones,
            infos,
        )

    def extract_logic_state(self, input_state):
        state = th.zeros((self.n_objects, self.n_features), dtype=th.int32)
        obj_count = {k: 0 for k in MAX_NB_OBJECTS.keys()}

        for obj in input_state:
            if obj.category not in self.relevant_objects:
                continue
            idx = self.obj_offsets[obj.category] + obj_count[obj.category]
            state[idx] = th.tensor([1, *obj.xy, obj.orientation])
            obj_count[obj.category] += 1
        return state

    def extract_neural_state(self, raw_input_state):
        return torch.tensor(
            [1 if obj.category == "Chicken" else 0 for obj in raw_input_state]
        )

    def close(self):
        for env in self.envs:
            env.close()
