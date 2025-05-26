import numpy as np


def reward_function(self) -> float:
    for obj in self.objects:
        if 'player' in str(obj).lower():
            player = obj
            break
    return round(self.org_reward / 1000.0, 2)