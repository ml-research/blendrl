from typing import Sequence
import torch
from blendrl.env_vectorized import VectorizedNudgeBaseEnv
from ocatari.core import OCAtari
import torch as th
from ocatari.ram.pong import MAX_NB_OBJECTS
import gymnasium as gym
import time
import numpy as np

from stable_baselines3.common.atari_wrappers import (  # isort:skip
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
    # env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)
    return env


class VectorizedNudgeEnv(VectorizedNudgeBaseEnv):
    """
    Vectorized NUDGE environment for Kangaroo.

    Args:
        mode (str): Mode of the environment. Possible values are "train" and "eval".
        n_envs (int): Number of environments.
        render_mode (str): Mode of rendering. Possible values are "rgb_array" and "human".
        render_oc_overlay (bool): Whether to render the overlay of OC.
        seed (int): Seed for the environment.
    """

    name = "pong"
    pred2action = {
        'noop': 0,
        'right': 2,
        'left': 3
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
        """
        Constructor for the VectorizedNudgeEnv class.

        Args:
            mode (str): Mode of the environment. Possible values are "train" and "eval".
            n_envs (int): Number of environments.
            render_mode (str): Mode of rendering. Possible values are "rgb_array" and "human".
            render_oc_overlay (bool): Whether to render the overlay of OC.
            seed (int): Seed for the environment.
        """
        super().__init__(mode)
        self.n_envs = n_envs
        # self.envs = [
        #     HackAtari(
        #         env_name="ALE/Pong-v5",
        #         mode="ram",
        #         obs_mode="ori",
        #         render_mode=render_mode,
        #         render_oc_overlay=render_oc_overlay
        #     )
        #     for i in range(n_envs)
        # ]
        # for i in range(n_envs):
        #     self.envs[i]._env = make_env(self.envs[i]._env)
        # initialize each HackAtari environment
        self.envs = [
            OCAtari(
                env_name="Pong-v4",
                mode="ram",
                obs_mode="ori",
                render_mode=render_mode,
                render_oc_overlay=render_oc_overlay,
            )
            for _ in range(n_envs)
        ]
        for i in range(n_envs):
            self.envs[i]._env = make_env(self.envs[i]._env)

        self.n_actions = len(self.pred2action)
        self.n_raw_actions = 6
        self.n_objects = 3
        self.n_features = 4
        self.seed = seed
        self.object_memory = {}

        # Compute index offsets. Needed to deal with multiple same-category objects
        self.obj_offsets = {}
        offset = 0
        for obj, max_count in MAX_NB_OBJECTS.items():
            self.obj_offsets[obj] = offset
            offset += max_count
        self.relevant_objects = set(MAX_NB_OBJECTS.keys())

    def reset(self):
        """
        Reset the environment.
        Returns:
            Tuple: Logic states and neural states for all environments.
        """
        logic_states = []
        neural_states = []
        seed_i = self.seed
        print("Env is being reset...")
        for env in self.envs:
            obs, state = env.reset(seed=seed_i)
            obs = torch.tensor(obs).float()
            state = env.objects
            raw_state = obs
            logic_state, neural_state = self.extract_logic_state(state), self.extract_neural_state(raw_state)
            logic_states.append(logic_state)
            neural_states.append(neural_state)
            seed_i += 1
        print("Env reset is done.")
        return torch.stack(logic_states), torch.stack(neural_states)

    def step(self, actions, is_mapped=False):
        """
        Perform a step in the environment.
        """
        assert (
                len(actions) == self.n_envs
        ), "Invalid number of actions: n_actions is {} and n_envs is {}".format(
            len(actions), self.n_envs
        )
        observations = []
        rewards = []
        truncations = []
        dones = []
        infos = []
        logic_states = []
        neural_states = []

        start = time.time()
        for i, env in enumerate(self.envs):
            action = actions[i]
            # Step in the environment
            obs, reward, truncation, done, info = env.step(action)
            raw_state = torch.tensor(obs).float()
            state = env.objects
            logic_state, neural_state = self.convert_state(state, raw_state)
            logic_states.append(logic_state)
            neural_states.append(neural_state)
            # Append results to lists
            observations.append(obs)
            rewards.append(reward)
            truncations.append(truncation)
            dones.append(done)
            infos.append(info)
        end = time.time()
        diff = end - start
        end = time.time()
        diff = end - start
        return (
            (torch.stack(logic_states), torch.stack(neural_states)),
            rewards,
            truncations,
            dones,
            infos,
        )

    def extract_logic_state1(self, raw_state):
        """
        Extracts the logic state from the input state using object_memory.
        """
        # state = torch.zeros((self.n_objects, self.n_features), dtype=torch.float32)
        # for idx, obj in enumerate(raw_state):
        #     if obj.category == "player":
        #         state[idx][0] = 1
        #     elif obj.category == "ball":
        #         state[idx][1] = 1
        #     elif "enemy" in obj.category:
        #         state[idx][2] = 1
        #
        #     current_x, current_y = obj.center
        #     # Retrieve previous coordinates from memory (if available)
        #     prev_x, prev_y = self.object_memory.get(obj.category, (current_x, current_y))
        #     # Store previous and current coordinates
        #     state[idx][-4:] = torch.tensor([prev_x, prev_y, current_x, current_y], dtype=torch.float32)
        #     # Update object_memory with current position
        #     self.object_memory[obj.category] = (current_x, current_y)
        logic_state = np.zeros((self.n_objects, self.n_features))

        for idx, obj in enumerate(raw_state):
            if obj.category == "player":
                logic_state[idx][0] = 1
            elif obj.category == "ball":
                logic_state[idx][1] = 1
            elif "enemy" in obj.category:
                logic_state[idx][2] = 1
            logic_state[idx][-4:] = np.array(obj.h_coords).flatten()
        return torch.tensor(logic_state, dtype=torch.float32)
        return state
    
    def extract_logic_state(self, raw_state):
        """
        Extracts the logic state from the input state using object memory.
        """
        obj_count = {k: 0 for k in MAX_NB_OBJECTS.keys()}
        state = th.zeros((self.n_objects, self.n_features), dtype=th.int32)

        for obj in raw_state:
            if obj.category not in self.relevant_objects:
                continue
            idx = self.obj_offsets[obj.category] + obj_count[obj.category]
            current_x, current_y = obj.center
            # Retrieve previous (x, y) from memory (if available)
            prev_x, prev_y = self.object_memory.get(obj.category, (current_x, current_y))
            # Store previous position before updating memory
            self.object_memory[obj.category] = (current_x, current_y)
            # Update state with previous and current coordinates from object memory
            state[idx] = th.tensor([prev_x, prev_y, current_x, current_y])
        return state


    def extract_neural_state(self, raw_input_state):
        """
        Extracts the neural state from the raw input state.
        Args:
            raw_input_state (torch.Tensor): Raw input state.
        Returns:
            torch.Tensor: Neural state.
        """
        return raw_input_state

    def close(self):
        """
        Close the environment.
        """
        for env in self.envs:
            env.close()
