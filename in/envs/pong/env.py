from typing import Sequence

import torch

from nudge.env import NudgeBaseEnv
from ocatari.core import OCAtari
import numpy as np


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

    def __init__(self, mode: str, render_mode="rgb_array", render_oc_overlay=False):
        super().__init__(mode)
        self.env = OCAtari(env_name="Pong", mode="ram",
                           render_mode=render_mode, render_oc_overlay=render_oc_overlay)

    def reset(self):
        self.env.reset()
        state = self.env.objects
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

        return torch.tensor(logic_state)

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
        return torch.flatten(self.extract_logic_state(raw_state))

    def close(self):
        self.env.close()