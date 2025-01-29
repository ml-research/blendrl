from typing import Sequence
import torch
from nudge.env import NudgeBaseEnv
from ocatari.core import OCAtari
import numpy as np
import gymnasium as gym
import torch as th
from ocatari.ram.pong import MAX_NB_OBJECTS
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)



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


def make_env_ori(env):
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.AutoResetWrapper(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    # env = gym.wrappers.ResizeObservation(env, (84, 84))
    # env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    return env


class NudgeEnv(NudgeBaseEnv):
    name = "pong"
    pred2action = {
        'noop': 0,
        'fire': 1,
        'right': 2,
        'left': 3,
        'rightfire': 4,
        'leftfire': 5,
    }
    pred_names: Sequence

    def __init__(self, mode: str, render_mode="rgb_array", render_oc_overlay=False, seed=None):
        super().__init__(mode)
        self.env = OCAtari(env_name="ALE/Pong-v5", mode="ram",obs_mode="ori",
                           render_mode=render_mode, render_oc_overlay=render_oc_overlay)

        self.env._env = make_env(self.env._env)
        self.n_actions = len(self.pred2action)
        self.n_raw_actions = 6
        self.n_objects = 3
        self.n_features = 4
        self.seed = seed

        # Compute index offsets. Needed to deal with multiple same-category objects
        self.obj_offsets = {}
        offset = 0
        for obj, max_count in MAX_NB_OBJECTS.items():
            self.obj_offsets[obj] = offset
            offset += max_count
        self.relevant_objects = set(MAX_NB_OBJECTS.keys())

    def reset(self):
        raw_state, _ = self.env.reset(seed=self.seed)
        state = self.env.objects
        logic_state, neural_state = self.extract_logic_state(state), self.extract_neural_state(raw_state)
        logic_state = logic_state.unsqueeze(0)
        return logic_state, neural_state

    def step(self, action, is_mapped: bool = False):
        raw_state, reward, truncations, done, infos = self.env.step(action)
        state = self.env.objects
        logic_state, neural_state = self.convert_state(state, raw_state)
        logic_state = logic_state.unsqueeze(0)
        return (logic_state, neural_state), reward, done, truncations, infos

    def extract_logic_state(self, raw_state):
        n_features = 4
        n_objects = 3
        logic_state = np.zeros((n_objects, n_features))
        for i, entity in enumerate(raw_state):
            if entity.category == "player":
                logic_state[i][0] = 1
            elif entity.category == 'ball':
                logic_state[i][1] = 1
            elif "enemy" in entity.category:
                logic_state[i][2] = 1
            logic_state[i][-4:] = np.array(entity.h_coords).flatten()
            return th.tensor(logic_state)

        # state = th.zeros((self.n_objects, self.n_features), dtype=th.int32)
        # # seve bboxes for exlanation rendering
        # self.bboxes = th.zeros((self.n_objects, 4), dtype=th.int32)
        #
        # obj_count = {k: 0 for k in MAX_NB_OBJECTS.keys()}
        #
        # for obj in raw_state:
        #     if obj.category not in self.relevant_objects:
        #         continue
        #     idx = self.obj_offsets[obj.category] + obj_count[obj.category]
        #
        #     orientation = (
        #         obj.orientation.value if obj.orientation is not None else 0
        #     )
        #     state[idx] = th.tensor([1, *obj.center, orientation])
        #     obj_count[obj.category] += 1
        #     self.bboxes[idx] = th.tensor(obj.xywh)
        # return state



    # def extract_neural_state(self, raw_state):
    #     #ppimport pdb;pdb.set_trace()
    #     neural_state = torch.zeros((3,4))
    #     for i, inst in enumerate(raw_state):
    #         if inst.category == "Player":
    #             neural_state[0] = torch.tensor(inst.h_coords).flatten()
    #        # elif inst.category == "Enemy":
    #        #    #  neural_state.append([0, 1, 0, 0] + list(inst.xy) + list(inst.prev_xy))
    #        #  elif "Ball" in inst.category:
    #        #      neural_state.append([0, 0, 1, 0] + list(inst.xy) + list(inst.prev_xy))
    #        #  #else:
    #            # neural_state.append([0, 0, 0, 1] + list(inst.xy) + list(inst.prev_xy))
    #
    #     # if len(neural_state) < 11:
    #     #    neural_state.extend([[0] * 6 for _ in range(11 - len(neural_state))])

        #return neural_state

    def extract_neural_state(self, raw_state):
        return torch.Tensor(raw_state).unsqueeze(0)

    def close(self):
        self.env.close()