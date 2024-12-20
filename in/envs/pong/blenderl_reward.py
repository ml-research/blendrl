import numpy as np


def reward_function(self) -> float:
    print (self.objects) 
    for obj in self.objects:
        if 'ball' in str(obj).lower():
            ball = obj
            break
    
    # Todo: When ball changes direction towards the enemy, give reward
    if self.org_reward == 1.0:
        reward = 1.0
    else:
        reward = 0.0
    return reward