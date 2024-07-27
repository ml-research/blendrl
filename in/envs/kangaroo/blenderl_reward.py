import numpy as np


def reward_function(self, game_reward) -> float:
    for obj in self.objects:
        if 'player' in str(obj).lower():
            player = obj
            break

    # got reawrd and previous step was on the top platform -> reached the child
    # if game_reward == 1.0 and player.prev_y == 4:
    #    reward = 10.0
    # x = 129
    if player.y == 4:
        reward = 0.2
    # BUG ↓ with multi envs, rewards collected repeatedly 
    # if player.y == 4 and player.prev_y != 4:
    #     reward = 1.0
    elif game_reward == 1.0 and player.prev_y != 4:
        reward = 0.05
    else:
        reward = 0.0
    return reward