import numpy as np

def reward_function(self) -> float:
    reward = 0.0
    for obj in self.objects:
        if 'player' in str(obj).lower():
            player = obj
            break

    # got reawrd and previous step was on the top platform -> reached the child
    # if game_reward == 1.0 and player.prev_y == 4:
    #    reward = 10.0
    # x = 129
    # if player.y == 4:
    # reward = 0.2
    # BUG ↓ with multi envs, rewards collected repeatedlydd
    if player.y == 4 and player.prev_y != 4:
        reward += 20.0  # Bonus for reaching the top platform

    # Reward for collecting fruits
    for obj in self.prev_objects:
        if 'fruit' in str(obj).lower() and obj not in self.objects:
            reward += 0.5  # Fruit collected

    # Reward for avoiding coconuts
    for obj in self.prev_objects:
        if 'coconut' in str(obj).lower() and obj in self.objects:
            coconut = next(o for o in self.objects if o == obj)
            if coconut.center[1] > player.center[1]:  # Coconut moved past the player
                reward += 2.0

    # Reward for punching monkeys or coconuts
    for obj in self.prev_objects:
        if ('monkey' in str(obj).lower() or 'coconut' in str(obj).lower()) and obj not in self.objects:
            reward += 1.0

    # Penalty for getting hit by a coconut or losing a life
    if player.lost_life:
        reward -= 10.0
    for obj in self.objects:
        if 'coconut' in str(obj).lower() and player.center == obj.center:
            reward -= 10.0


    # Ensure lives do not drop below 0
    self.lives = max(self.lives, 0)

    return reward
