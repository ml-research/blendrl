import numpy as np


def reward_function(self) -> float:
    # Find the player object
    for obj in self.objects:
        if "player" in str(obj).lower():
            player = obj
            break

    # Custom reward logic for River Raid
    if self.org_reward > 0:  # Reward for destroying an enemy or collecting fuel
        reward = self.org_reward * 10.0
    elif player.x > player.prev_x:  # Reward for moving forward in the river
        reward = 1.0
    else:  # No reward for other actions
        reward = 0.0

    return reward
