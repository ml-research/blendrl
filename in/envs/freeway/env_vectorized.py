import time
from typing import Sequence
import torch
from blendrl.env_vectorized import VectorizedNudgeBaseEnv
from hackatari.core import HackAtari
import torch as th
from ocatari.ram.freeway import MAX_NB_OBJECTS
import gymnasium as gym

import time

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


class VectorizedNudgeEnv(VectorizedNudgeBaseEnv):
    """
    Vectorized NUDGE environment for Freeway.

    Args:
        mode (str): Mode of the environment. Possible values are "train" and "eval".
        n_envs (int): Number of environments.
        render_mode (str): Mode of rendering. Possible values are "rgb_array" and "human".
        render_oc_overlay (bool): Whether to render the overlay of OC.
        seed (int): Seed for the environment.
    """

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
        """
        Constructor for the VectorizedNudgeEnv class.

        Args:
            mode (str): Mode of the environment. Possible values are "train" and "eval".
            n_envs (int): Number of environments.
            render_mode (str): Mode of rendering. Possible values are "rgb_array" and "human".
            render_oc_overlay (bool): Whether to render the overlay of OC.
            seed (int): Seed for the environment.
        """
        print(mode)
        super().__init__(mode)
        # set up multiple envs
        self.n_envs = n_envs
        # initialize each HackAtari environment
        self.envs = [
            HackAtari(
                env_name="ALE/Freeway-v5",
                mode="ram",
                obs_mode="ori",
                modifs=[],
                #rewardfunc_path="in/envs/freeway/blenderl_reward.py",
                render_mode=render_mode,
                render_oc_overlay=render_oc_overlay,
            )
            for i in range(n_envs)
        ]
        # apply wrapper to _env
        for i in range(n_envs):
            self.envs[i]._env = make_env(self.envs[i]._env)

        self.n_actions = 3
        # self.n_raw_actions = 18
        self.n_objects = 12
        self.n_features = 6
        self.seed = seed

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
            logic_states (torch.Tensor): Logic states.
            neural_states (torch.Tensor): Neural states.
        """
        logic_states = []
        neural_states = []
        seed_i = self.seed
        print("Env is being reset...")
        for env in self.envs:
            obs, _ = env.reset(seed=seed_i)
            # lazy frame to tensor
            obs = torch.tensor(obs).float()
            state = env.objects
            raw_state = obs  # self.env.dqn_obs
            logic_state, neural_state = self.extract_logic_state(
                state
            ), self.extract_neural_state(raw_state)
            logic_states.append(logic_state)
            neural_states.append(neural_state)
            seed_i += 1
        print("Env reset is done.")
        return torch.stack(logic_states), torch.stack(neural_states)

    def step(self, actions, is_mapped=False):
        """
        Perform a step in the environment.

        Args:
            actions (torch.Tensor): Actions to be performed in the environment.
            is_mapped (bool): Whether the actions are already mapped.
        Returns:
            Tuple: Tuple containing:
                - torch.Tensor: Observations.
                - list: Rewards.
                - list: Truncations.
                - list: Dones.
                - list: Infos.
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
            # make a step in the env
            obs, reward, truncation, done, info = env.step(action)
            # Check with multiple envs
            # for obj in env.objects:
            #     if "Player" in str(obj):
            #         print("Env_{}".format(i), obj)
            # if reward > 0.5:
            #     print("Reward: ", reward)
            # lazy frame to tensor
            raw_state = torch.tensor(obs).float()
            # get logic and neural state
            state = env.objects
            logic_state, neural_state = self.convert_state(state, raw_state)
            logic_states.append(logic_state)
            neural_states.append(neural_state)
            observations.append(obs)
            rewards.append(reward)
            truncations.append(truncation)
            dones.append(done)
            infos.append(info)
        end = time.time()
        diff = end - start
        # print("Time taken for step: ", diff)

        end = time.time()
        diff = end - start
        # print("Time taken for step: ", diff)

        # observations = torch.stack(observations)
        return (
            (torch.stack(logic_states), torch.stack(neural_states)),
            rewards,
            truncations,
            dones,
            infos,
        )

    def extract_logic_state(self, raw_state):
        """
        Extracts the logic state from the input state.
        Args:
            raw_state (list): List of objects in the environment.
        Returns:
            torch.Tensor: Logic state with 6 features per object:
            - visibility (1/0)
            - pixel_x
            - pixel_y
            - grid_x
            - grid_y
            - type (1=chicken, 2=car)
            
            Comment:
            in ocatari/ram/freeway.py:
                MAX_NB_OBJECTS = {
                    'Chicken': 2,
                    'Car': 10}
        """
        state = th.zeros((self.n_objects, self.n_features), dtype=th.int32)

        obj_count = {k: 0 for k in MAX_NB_OBJECTS.keys()}

        for obj in raw_state:
            print(f"Processing object: {obj}")  # Debug print
            if obj.category not in self.relevant_objects:
                continue
            
            idx = self.obj_offsets[obj.category] + obj_count[obj.category]
            
            # Get both coordinate systems
            pixel_x, pixel_y = obj.center  # Pixel coordinates
            grid_x, grid_y = obj.xy if hasattr(obj, 'xy') else (0, 0)  # Grid coordinates
            
            # Create 6-feature vector
            obj_type = 1 if obj.category == 'Chicken' else 2
            state[idx] = th.tensor([
                1,           # visibility
                pixel_x,     # pixel x-coordinate
                pixel_y,     # pixel y-coordinate
                grid_x,      # grid x-coordinate
                grid_y,      # grid y-coordinate
                obj_type     # object type
            ])
            
            obj_count[obj.category] += 1

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
