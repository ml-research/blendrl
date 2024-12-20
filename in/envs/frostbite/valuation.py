import torch as th

from nsfr.utils.common import bool_to_probs

def low_temperature(temperature: th.Tensor) -> th.Tensor:
    """True iff temperature is below 15/45."""
    result = temperature[..., 1] < 15
    return bool_to_probs(result)

def left_of_bear(player: th.Tensor, bear: th.Tensor) -> th.Tensor:
    """True iff the player is 'left of' the bear."""
    player_x = player[..., 1]
    bear_x = bear[..., 7]
    bear_prob = bear[:, 0]
    return bool_to_probs(player_x < bear_x) * bear_prob

def right_of_bear(player: th.Tensor, bear: th.Tensor) -> th.Tensor:
    """True iff the player is 'right of' the bear."""
    player_x = player[..., 1]
    bear_x = bear[..., 7]
    bear_prob = bear[:, 0]
    return bool_to_probs(player_x > bear_x) * bear_prob

def left_of_door(player: th.Tensor, door: th.Tensor) -> th.Tensor:
    """True iff the player is 'left of' the door."""
    player_x = player[..., 1]
    door_x = door[..., 1]
    door_prob = door[:, 0]
    return bool_to_probs(player_x < door_x) * door_prob

def right_of_door(player: th.Tensor, door: th.Tensor) -> th.Tensor:
    """True iff the player is 'right of' the door."""
    player_x = player[..., 1]
    door_x = door[..., 1]
    door_prob = door[:, 0]
    return bool_to_probs(player_x > door_x) * door_prob