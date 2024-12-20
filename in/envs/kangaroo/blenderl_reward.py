import numpy as np


def reward_function(self) -> float:
    for obj in self.objects:
        if 'player' in str(obj).lower():
            player = obj
            break

    # player reaches top platform 
    if player.y == 4 and player.prev_y != 4:
        reward = 20.0
    # player kills enemy
    elif self.org_reward == 1.0 and player.prev_y != 4:
        reward = 1.0
    else:
        reward = 0.0
    return reward