from typing import Sequence
import torch
from blendrl.env_vectorized import VectorizedNudgeBaseEnv
from hackatari.core import HackAtari
import torch as th
from ocatari.ram.donkeykong import MAX_ESSENTIAL_OBJECTS
import gymnasium as gym
from ocatari.core import OCAtari

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
    # env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    return env


class VectorizedNudgeEnv(VectorizedNudgeBaseEnv):
    """
    Vectorized NUDGE environment for DonkeyKong.
    
    Args:
        mode (str): Mode of the environment. Possible values are "train" and "eval".
        n_envs (int): Number of environments.
        render_mode (str): Mode of rendering. Possible values are "rgb_array" and "human".
        render_oc_overlay (bool): Whether to render the overlay of OC.
        seed (int): Seed for the environment.
    """
    name = "donkeykong"
    # MAX_ESSENTIAL_OBJECTS = {
    #     'Player': 1, "DonkeyKong": 1, "Girlfriend": 1, "Hammer": 1, "Barrel": 6, "Ladder": 10
    # }
    pred2action = {
        'noop': 0,
        'fire': 1,
        'up': 2,
        'right': 3,
        'left': 4,
        'down': 5,
        'up_right': 6,
        'up_left': 7,
        'fire_right': 11,
        'fire_left': 12
    }
    pred_names: Sequence

    def __init__(self, mode: str, n_envs: int, render_mode="rgb_array", render_oc_overlay=False, seed=None):
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
        # set up multiple envs
        self.n_envs = n_envs
        self.envs = [
            OCAtari(
                env_name="ALE/DonkeyKong-v5",
                mode="ram",
                obs_mode="ori",
                render_mode=render_mode,
                render_oc_overlay=render_oc_overlay,
            )
            for _ in range(n_envs)
        ]
        # apply wrapper to _env
        for i in range(n_envs):
            self.envs[i]._env = make_env(self.envs[i]._env)

        self.n_actions = len(self.pred2action)
        self.n_raw_actions = 18
        self.n_objects = 20
        self.n_features = 4  # visible, x-pos, y-pos, right-facing
        self.seed = seed

        # Compute index offsets. Needed to deal with multiple same-category objects
        self.obj_offsets = {}
        offset = 0
        for (obj, max_count) in MAX_ESSENTIAL_OBJECTS.items():
            self.obj_offsets[obj] = offset
            offset += max_count
        self.relevant_objects = set(MAX_ESSENTIAL_OBJECTS.keys())

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
            logic_state, neural_state = self.extract_logic_state(state), self.extract_neural_state(raw_state)
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
        assert len(actions) == self.n_envs, "Invalid number of actions: n_actions is {} and n_envs is {}".format(
            len(actions), self.n_envs)
        observations = []
        rewards = []
        truncations = []
        dones = []
        infos = []
        logic_states = []
        neural_states = []

        # start = time.time()        
        for i, env in enumerate(self.envs):
            action = actions[i]
            # make a step in the env
            obs, reward, truncation, done, info = env.step(action)
            raw_state = torch.tensor(obs).float()
            state = env.objects
            logic_state, neural_state = self.convert_state(state, raw_state)
            logic_states.append(logic_state)
            neural_states.append(neural_state)
            observations.append(obs)
            rewards.append(reward)
            truncations.append(truncation)
            dones.append(done)
            infos.append(info)

        return (torch.stack(logic_states), torch.stack(neural_states)), rewards, truncations, dones, infos

    def extract_logic_state(self, input_state):
        """ 
        Extracts the logic state from the input state.
        Args:
            input_state (list): List of objects in the environment.
        Returns:
            torch.Tensor: Logic state.
        """
        state = th.zeros((self.n_objects, self.n_features), dtype=th.int32)

        obj_count = {k: 0 for k in MAX_ESSENTIAL_OBJECTS.keys()}

        for obj in input_state:
            if obj.category not in self.relevant_objects:
                continue
            idx = self.obj_offsets[obj.category] + obj_count[obj.category]
            if obj.category == "Time":
                state[idx] = th.Tensor([1, obj.value, 0, 0])
            else:
                orientation = obj.orientation.value if obj.orientation is not None else 0
                state[idx] = th.tensor([1, *obj.center, orientation])
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
