import numpy as np


def reward_function(self) -> float:
    for obj in self.objects:
        if 'player' in str(obj).lower():
            player = obj
            break

    # BUG ↓ with multi envs, rewards collected repeatedlydd
    if self.org_reward == 1.0 and player.y != 45:
        # kill an enemy, e.g. a shark
        reward = 0.5
    elif self.org_reward == 1.0 and player.y == 45:
        # rescue 6 divers ('player.y == 45' means you're at the water surface)
        reward = 1.0
    else:
        reward = 0.0
    return reward