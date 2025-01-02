from typing import Sequence
import torch
from nudge.env import NudgeBaseEnv
from hackatari.core import HackAtari
import numpy as np
import torch as th
from ocatari.ram.seaquest import MAX_ESSENTIAL_OBJECTS
import gymnasium as gym



from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

def make_env(env):
    # def thunk():
        # if capture_video and idx == 0:
            # env = gym.make(env_id, render_mode="rgb_array")
            # env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # else:
            # env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    env = gym.wrappers.AutoResetWrapper(env)
    return env



class NudgeEnv(NudgeBaseEnv):
    name = "tennis"
    pred2action = {
        'noop': 0,
        'fire': 1,
        'up': 2,
        'right': 3,
        'left': 4,
        'down': 5,
    }


    pred_names: Sequence

    def __init__(self, mode: str, render_mode="rgb_array", render_oc_overlay=False):
        super().__init__(mode)
        self.env = HackAtari(env_name="Pong", mode="ram",
                           render_mode=render_mode, render_oc_overlay=render_oc_overlay)

    def reset(self):
        self.env.reset()
        state = self.env.objects
        print(state)
        return self.convert_state(state)

    def step(self, action, is_mapped: bool = False):
        if not is_mapped:
            action = self.map_action(action)
        _, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        state = self.env.objects
        return self.convert_state(state), reward, done

    def extract_logic_state(self, raw_state):
        n_features = 4
        n_objects = 4
        logic_state = np.zeros((n_objects, n_features))
        for i, entity in enumerate(raw_state):
            if entity.category == "player":
                logic_state[i][0] = 1
            elif entity.category == 'ball':
                logic_state[i][1] = 1
            elif "enemy" in entity.category:
                logic_state[i][2] = 1
            elif "ballshadow" in entity.category:
                logic_state[i][3] = 1
            logic_state[i][-5:] = np.array(entity.h_coords).flatten()

        return torch.tensor(logic_state)

    def extract_neural_state(self, raw_state):
        return torch.flatten(self.extract_logic_state(raw_state))

    def close(self):
        self.env.close()