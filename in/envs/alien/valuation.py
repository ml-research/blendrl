import torch as th
from nsfr.utils.common import bool_to_probs

def visible_alien(alien: th.Tensor) -> th.Tensor:
    return bool_to_probs(alien[..., 0] == 1)

def visible_egg(egg: th.Tensor) -> th.Tensor:
    return bool_to_probs(egg[..., 0] == 1)

def visible_pulsar(pulsar: th.Tensor) -> th.Tensor:
    return bool_to_probs(pulsar[..., 0] == 1)

def visible_rocket(rocket: th.Tensor) -> th.Tensor:
    return bool_to_probs(rocket[..., 0] == 1)

def facing_left(player: th.Tensor) -> th.Tensor:
    return bool_to_probs(player[..., 3] == 0)

def facing_right(player: th.Tensor) -> th.Tensor:
    return bool_to_probs(player[..., 3] == 1)

def left_of_alien(player: th.Tensor, alien: th.Tensor) -> th.Tensor:
    return bool_to_probs(player[..., 1] < alien[..., 1])

def left_of_egg(player: th.Tensor, egg: th.Tensor) -> th.Tensor:
    return bool_to_probs(player[..., 1] < egg[..., 1])

def right_of_alien(player: th.Tensor, alien: th.Tensor) -> th.Tensor:
    return bool_to_probs(player[..., 1] > alien[..., 1])

def right_of_egg(player: th.Tensor, egg: th.Tensor) -> th.Tensor:
    return bool_to_probs(player[..., 1] > egg[..., 1])

def close_by_alien(player: th.Tensor, alien: th.Tensor) -> th.Tensor:
    return _close_by(player, alien, threshold=40)

def close_by_egg(player: th.Tensor, egg: th.Tensor) -> th.Tensor:
    return _close_by(player, egg, threshold=20)

def close_by_pulsar(player: th.Tensor, pulsar: th.Tensor) -> th.Tensor:
    return _close_by(player, pulsar, threshold=30)

def close_by_rocket(player: th.Tensor, rocket: th.Tensor) -> th.Tensor:
    return _close_by(player, rocket, threshold=35)

def higher_than_alien(player: th.Tensor, alien: th.Tensor) -> th.Tensor:
    return bool_to_probs(player[..., 2] < alien[..., 2])

def higher_than_egg(player: th.Tensor, egg: th.Tensor) -> th.Tensor:
    return bool_to_probs(player[..., 2] < egg[..., 2])

def lower_than_alien(player: th.Tensor, alien: th.Tensor) -> th.Tensor:
    return bool_to_probs(player[..., 2] > alien[..., 2])

def lower_than_egg(player: th.Tensor, egg: th.Tensor) -> th.Tensor:
    return bool_to_probs(player[..., 2] > egg[..., 2])

def same_level_alien(player: th.Tensor, alien: th.Tensor) -> th.Tensor:
    return bool_to_probs(abs(player[..., 2] - alien[..., 2]) < 5)

def same_level_egg(player: th.Tensor, egg: th.Tensor) -> th.Tensor:
    return bool_to_probs(abs(player[..., 2] - egg[..., 2]) < 5)

def low_lives(life: th.Tensor) -> th.Tensor:
    return bool_to_probs(life[..., 1] < 2)

def all_eggs_destroyed(eggs: th.Tensor) -> th.Tensor:
    return bool_to_probs(th.all(eggs[..., 0] == 0))

def some_eggs_remaining(eggs: th.Tensor) -> th.Tensor:
    return bool_to_probs(th.any(eggs[..., 0] == 1))

def many_aliens(aliens: th.Tensor) -> th.Tensor:
    return bool_to_probs(th.sum(aliens[..., 0]) > 5)

def _close_by(player: th.Tensor, obj: th.Tensor, threshold=32) -> th.Tensor:
    player_x, player_y = player[..., 1], player[..., 2]
    obj_x, obj_y = obj[..., 1], obj[..., 2]
    obj_prob = obj[:, 0]
    distance = ((player_x - obj_x) ** 2 + (player_y - obj_y) ** 2).sqrt()
    return bool_to_probs(distance < threshold) * obj_prob

def test_predicate_global(global_state: th.Tensor) -> th.Tensor:
    return bool_to_probs(global_state[..., 0, 2] < 100)

def test_predicate_object(agent: th.Tensor) -> th.Tensor:
    return bool_to_probs(agent[..., 2] < 100)

def true_predicate(agent: th.Tensor) -> th.Tensor:
    return bool_to_probs(th.tensor([True]))

def false_predicate(agent: th.Tensor) -> th.Tensor:
    return bool_to_probs(th.tensor([False]))
