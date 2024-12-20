import numpy as np


def reward_function(self) -> float:
    for obj in self.objects:
        if 'player' in str(obj).lower():
            player = obj
            break

    if self.org_reward == 1.0 and player.y != 45:
        reward = 1.0
    else:
        reward = 0.0
    #great reward for completing stage
    #small reward for collecting fish
    #medium reward for high remaining temperature
    #tiny reward for collecting iceblock
    #small penalty for switching direction?
    #great penalty for dying
    return reward