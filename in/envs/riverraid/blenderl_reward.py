import numpy as np


def reward_function(self) -> float:
    # Find the player object
    player = next((obj for obj in self.objects if "player" in str(obj).lower()), None)

    if player is None:
        return 0.0  # No reward if player is not found

    # Reward logic for River Raid
    if self.org_reward > 0:
        reward = (
            self.org_reward * 10.0
        )  # Higher reward for shooting enemies or collecting fuel
    elif player.x > player.prev_x:
        reward = 1.0  # Small reward for forward movement
    else:
        reward = -0.1  # Penalize stagnation or backward movement

    return reward
