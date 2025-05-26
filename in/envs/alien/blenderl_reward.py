import numpy as np

def reward_function(self) -> float:

    for obj in self.objects:
        if "player" in str(obj).lower():
            player = obj
            break

    reward = 0.0

    if self.org_reward == 1.0:
        # Reward for eliminating an alien
        reward = 0.5

    for obj in self.objects:
        if "egg" in str(obj).lower() and obj.xy == player.xy:
            # Reward for collecting an egg
            reward = 1.0
            break
        if "pulsar" in str(obj).lower() and obj.xy == player.xy:
            # Small reward for interacting with a pulsar
            reward = 0.2
            break

    return reward
