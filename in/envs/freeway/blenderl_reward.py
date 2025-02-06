import numpy as np


def reward_function(self) -> float:
    """
    Reward function for Freeway:
    - +20.0 for reaching the top (crossing the road successfully)
    - -10.0 for getting hit by a car
    - +0.01 for moving upward
    - -0.01 for moving downward
    """
    reward = 0.0
    
    # Get the chicken's position (should be the first chicken in objects)
    for obj in self.objects:
        if 'player' in str(obj).lower():
            player = obj
            break
    
    if player is None:
        raise ValueError("Player not found")
    
    # Check if chicken reached the top (main reward)
    if player.xy[1] == 0:  # Top of the screen in grid coordinates
        reward += 20.0
    
    # Check for collision with cars (penalty)
    for obj in self.objects:
        if obj.category == "Car":
            if self.check_collision(player, obj):
                reward -= 10.0
                break
    
    # Reward for vertical movement
    if hasattr(self, '_prev_y'):
        y_diff = self._prev_y - player.xy[1]
        if y_diff > 0:  # Moving up
            reward += 0.01
        elif y_diff < 0:  # Moving down
            reward -= 0.01
    
    # Store current y position for next frame
    self._prev_y = player.xy[1]
    
    return reward

def check_collision(self, player, car):
    """Helper function to check if chicken collides with a car"""
    # Using grid coordinates for collision detection
    return (player.xy[1] == car.xy[1] and  # Same row
            abs(player.xy[0] - car.xy[0]) < 1)  # Close in x-axis