import numpy as np


def reward_function(self) -> float:
    """
    Reward function for the Freeway environment.
    """
    player = None
    for obj in self.objects:
        if "chicken" in str(obj).lower():
            player = obj
            break

    if player is None:
        # if no player object is found, return 0 reward
        return 0.0

    # reward for moving closer to the top
    if player.prev_y > player.y:
        # Progress upward
        reward = 1.0
    elif player.prev_y < player.y:
        # moving downward (penalty)
        reward = -0.5
    else:
        # staying in the same position
        reward = 0.0

    # collision with a car
    for obj in self.objects:
        if "car" in str(obj).lower() and obj.collides_with(player):
            reward = -5.0
            break

    # reward for successfully crossing the road (reaching the top platform)
    if player.y == 0:
        reward = 10.0

    return reward
