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
            reward += 3.0

    # Reward for ringing the bell
    for obj in self.objects:
        if 'bell' in str(obj).lower() and self.is_player_close(obj) and player.center[1] < obj.center[1]:
            reward += 5.0  # Ringing the bell adds a significant bonus

    # Punishment for being hit by falling coconuts
    for obj in self.objects:
        if 'fallingcoconut' in str(obj).lower() and player.center == obj.center:
            reward -= 15.0
            self.lives -= 1

    # Punishment for being hit by thrown coconuts
    for obj in self.objects:
        if 'throwncoconut' in str(obj).lower() and player.center == obj.center:
            reward -= 15.0
            self.lives -= 1

    # Punishment for being hit by a monkey
    for obj in self.objects:
        if 'monkey' in str(obj).lower() and player.center == obj.center:
            reward -= 20.0
            self.lives -= 1

    # Reward for avoiding falling coconuts
    for obj in self.prev_objects:
        if 'fallingcoconut' in str(obj).lower() and obj in self.objects:
            coconut = next(o for o in self.objects if o == obj)
            if coconut.center[1] > player.center[1]:
                reward += 2.0

    # Reward for avoiding thrown coconuts
    for obj in self.prev_objects:
        if 'throwncoconut' in str(obj).lower() and obj in self.objects:
            coconut = next(o for o in self.objects if o == obj)
            if coconut.center[0] > player.center[0]:
                reward += 2.0

    # Reward for punching monkeys
    for obj in self.prev_objects:
        if 'monkey' in str(obj).lower() and obj not in self.objects:
            reward += 5.0

    # Reward for punching coconuts (falling or thrown)
    for obj in self.prev_objects:
        if 'coconut' in str(obj).lower() and obj not in self.objects:
            reward += 3.0

    # Reward for gaining extra life
    current_lives = self.lives
    for obj in self.objects:
        if 'life' in str(obj).lower() and obj not in self.prev_objects:
            current_lives += 1
            reward += 10.0

    self.lives = current_lives
    self.lives = max(self.lives, 0)

    return reward
