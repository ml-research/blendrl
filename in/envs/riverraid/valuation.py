import torch as th

from nsfr.utils.common import bool_to_probs

# MAX_ESSENTIAL_OBJECTS for River Raid:
# 'Player': 1, (0)
# 'Fuel': 1, (1)
# 'Bridge': 1, (2)
# 'EnemyBoat': 3, (3)
# 'EnemyHelicopter': 2, (6)
# 'Land': 10, (8)
# 'Bullet': 5, (18)


def nothing_around(objs: th.Tensor) -> th.Tensor:
    # Target objects: enemy boats, helicopters, bullets
    enemy_boats = objs[:, 3:6]
    enemy_helicopters = objs[:, 6:8]
    bullets = objs[:, 18:23]

    target_objs = th.cat([enemy_boats, enemy_helicopters, bullets], dim=1)
    players = objs[:, 0].unsqueeze(1).expand(-1, target_objs.size(1), -1)

    probs = th.stack(
        [
            _close_by(players[:, i, :], target_objs[:, i, :])
            for i in range(target_objs.size(1))
        ],
        dim=1,
    )
    max_closeby_prob, _ = probs.max(dim=1)
    return (1.0 - max_closeby_prob).float()


def close_by_fuel(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _close_by(player, obj)


def close_by_bridge(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _close_by(player, obj)


def close_by_helicopter(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _close_by(player, obj)


def close_by_bullet(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _close_by(player, obj)


def close_by_enemy_base(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _close_by(player, obj)


def close_by_enemy_ship(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _close_by(player, obj)


def left_of_bridge(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_x = player[..., 1]
    obj_x = obj[..., 1]
    obj_prob = obj[:, 0]
    return bool_to_probs(player_x < obj_x) * obj_prob


def right_of_bridge(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_x = player[..., 1]
    obj_x = obj[..., 1]
    obj_prob = obj[:, 0]
    return bool_to_probs(player_x > obj_x) * obj_prob


def deeper_than_enemy(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    obj_prob = obj[:, 0]
    return bool_to_probs(player_y > obj_y + 4) * obj_prob


def higher_than_enemy(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    obj_prob = obj[:, 0]
    return bool_to_probs(player_y < obj_y - 4) * obj_prob


def fuel_low(fuel_bar: th.Tensor) -> th.Tensor:
    return bool_to_probs(fuel_bar[..., 1] < 16)


def _close_by(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    threshold = 32
    player_x = player[:, 1]
    player_y = player[:, 2]
    obj_x = obj[:, 1]
    obj_y = obj[:, 2]
    obj_prob = obj[:, 0]
    x_dist = (player_x - obj_x).pow(2)
    y_dist = (player_y - obj_y).pow(2)
    dist = (x_dist + y_dist).sqrt()
    return bool_to_probs(dist < threshold) * obj_prob


def left_of_river(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True iff the player is 'left of' the object."""
    player_x = player[..., 1]
    obj_x = obj[..., 1]
    obj_prob = obj[:, 0]
    return bool_to_probs(player_x < obj_x) * obj_prob


def right_of_river(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True iff the player is 'right of' the object."""
    player_x = player[..., 1]
    obj_x = obj[..., 1]
    obj_prob = obj[:, 0]
    return bool_to_probs(player_x > obj_x) * obj_prob


# ---------------------------------subject to change---------------------------------#
def on_bridge_player(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    obj_prob = obj[:, 0]
    return bool_to_probs(player_y == obj_y) * obj_prob


def on_bridge_player(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    obj_prob = obj[:, 0]
    return bool_to_probs(player_y == obj_y) * obj_prob


def on_bridge_river(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    obj_prob = obj[:, 0]
    return bool_to_probs(player_y == obj_y) * obj_prob


def on_river(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    obj_prob = obj[:, 0]
    return bool_to_probs(player_y == obj_y) * obj_prob


def same_level_river(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    obj_prob = obj[:, 0]
    return bool_to_probs(player_y == obj_y) * obj_prob


# ----------------------------------------------------------------------------------#


def test_predicate_global(global_state: th.Tensor) -> th.Tensor:
    result = global_state[..., 0, 2] < 100
    return bool_to_probs(result)


def test_predicate_object(agent: th.Tensor) -> th.Tensor:
    result = agent[..., 2] < 100
    return bool_to_probs(result)


def true_predicate(agent: th.Tensor) -> th.Tensor:
    return bool_to_probs(th.tensor([True]))


def false_predicate(agent: th.Tensor) -> th.Tensor:
    return bool_to_probs(th.tensor([False]))
