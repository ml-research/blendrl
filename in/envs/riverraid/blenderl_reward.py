import numpy as np


def reward_function(self) -> float:
    # Find the player object
    for obj in self.objects:
        if "player" in str(obj).lower():
            player = obj
            break

    # reward logic for River Raid
    if self.org_reward > 0:  
        reward = self.org_reward * 10.0
    elif player.x > player.prev_x:  
        reward = 1.0
    else:  
        reward = 0.0

    return reward
