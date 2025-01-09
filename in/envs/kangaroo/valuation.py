import torch as th

from nsfr.utils.common import bool_to_probs

""" in ocatari/ram/kangaroo.py :
        MAX_ESSENTIAL_OBJECTS = {
            'Player': 1, (0)
            'Child': 1, (1)
            'Fruit': 3, (2)
            'Bell': 1, (5)
            'Platform': 20,
            'Ladder': 6,
            'Monkey': 4,
            'FallingCoconut': 1,
            'ThrownCoconut': 3,
            'Life': 8,
            'Time': 1,}       
"""


def nothing_around(objs: th.Tensor) -> th.Tensor:
    # target objects: fruit, bell, monkey, fallingcoconut, throwncoconut
    fruits = objs[:, 2:5]
    bell = objs[:, 5].unsqueeze(1)
    monkey = objs[:, 32:36]
    falling_coconut = objs[:, 36].unsqueeze(1)
    thrown_coconut = objs[:, 37:40]
    # target_objs = th.cat([fruits, bell, monkey, falling_coconut, thrown_coconut], dim=1)
    target_objs = th.cat([monkey, falling_coconut, thrown_coconut], dim=1)
    players = objs[:, 0].unsqueeze(1).expand(-1, target_objs.size(1), -1)

    # batch_size * num_target_objs
    probs = th.stack([_close_by(players[:, i, :], target_objs[:, i, :]) for i in range(target_objs.size(1))], dim=1)

    max_closeby_prob, _ = probs.max(dim=1)
    result = (1.0 - max_closeby_prob).float()
    return result


def _in_field(obj: th.Tensor) -> th.Tensor:
    x = obj[..., 1]
    prob = obj[:, 0]
    return bool_to_probs(th.logical_and(16 <= x, x <= 16 + 128)) * prob


# ---------------------------------------------------------------------------------
# platform logic
def _on_platform(obj1: th.Tensor, obj2: th.Tensor) -> th.Tensor:
    """True iff obj1 is 'on' obj2."""
    obj1_y = obj1[..., 2]
    obj2_y = obj2[..., 2]
    obj1_prob = obj1[:, 0]
    obj2_prob = obj2[:, 0]
    result = th.logical_and(12 < obj2_y - obj1_y, obj2_y - obj1_y < 60)
    return bool_to_probs(result) * obj1_prob * obj2_prob


def on_pl_player(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _on_platform(player, obj)


def on_pl_ladder(ladder: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _on_platform(ladder, obj)


def on_ladder(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_x = player[..., 1]
    obj_x = obj[..., 1]
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    obj_prob = obj[:, 0]
    x_prob = bool_to_probs(abs(player_x - obj_x) < 4)
    return x_prob
    # y_prob = bool_to_probs(obj_y > player_y - 8)
    # return  x_prob * y_prob * obj_prob * same_level_ladder(player, obj)


def obj_below(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """
    Check if there is a platform directly below the player.
    """
    player_x, player_y = player[..., 1], player[..., 2]
    platform_x, platform_y = obj[..., 1], obj[..., 2]
    platform_prob = obj[:, 0]

    # Platform below and aligned horizontally
    is_below = (platform_y > player_y) & (platform_y - player_y < 60)
    is_aligned = abs(player_x - platform_x) < 5

    return bool_to_probs(is_below & is_aligned) * platform_prob


def obj_on_platform_below(player: th.Tensor, fruit: th.Tensor, bell: th.Tensor, platform: th.Tensor) -> th.Tensor:
    """
    Check if a fruit or bell is on the platform below the player.
    """
    fruit_below = obj_below(player, fruit) & _on_platform(fruit, platform)
    bell_below = obj_below(player, bell) & _on_platform(bell, platform)
    return fruit_below | bell_below


def safe_to_move(player: th.Tensor, monkey: th.Tensor, falling_coconut: th.Tensor, thrown_coconut: th.Tensor,
                 platform: th.Tensor) -> th.Tensor:
    """
    Check if it is safe for the player to move down (no monkeys or coconuts on the platform below).
    """
    hazards_below = (
            obj_below(player, monkey) |
            obj_below(player, falling_coconut) |
            obj_below(player, thrown_coconut)
    )
    hazards_on_platform = (
            _on_platform(monkey, platform) |
            _on_platform(falling_coconut, platform) |
            _on_platform(thrown_coconut, platform)
    )
    return ~bool_to_probs(hazards_below | hazards_on_platform)


def at_window_edge_right(player: th.Tensor, edge_tolerance: float = 5.0) -> th.Tensor:
    window_width = 160.0
    player_x = player[..., 1]
    at_right_edge = (player_x >= window_width - edge_tolerance)
    return bool_to_probs(at_right_edge)


def at_window_edge_left(player: th.Tensor, edge_tolerance: float = 5.0) -> th.Tensor:
    window_width = 160.0
    player_x = player[..., 1]
    at_left_edge = (player_x <= edge_tolerance)
    return bool_to_probs(at_left_edge)

def same_level_ladder(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    obj1_y = player[..., 2] + 10
    obj2_y = obj[..., 2]

    is_3rd_level = th.logical_and(th.logical_and(28 < obj1_y, obj1_y < 76), th.logical_and(28 < obj2_y, obj2_y < 76))
    is_2nd_level = th.logical_and(th.logical_and(76 < obj1_y, obj1_y < 124), th.logical_and(76 < obj2_y, obj2_y < 124))
    is_1st_level = th.logical_and(th.logical_and(124 < obj1_y, obj1_y < 172),
                                  th.logical_and(124 < obj2_y, obj2_y < 172))

    is_same_level = th.logical_or(is_3rd_level, th.logical_or(is_2nd_level, is_1st_level))
    return bool_to_probs(is_same_level)


# ---------------------------------------------------------------------------------
# fruit logic
def close_by_fruit(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _close_by(player, obj) * same_level(player, obj)


def on_pl_fruit(fruit: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _on_platform(fruit, obj)


# ---------------------------------------------------------------------------------
# bell logic
def same_level_bell(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return same_level(player, obj)


def on_pl_bell(bell: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _on_platform(bell, obj)


def close_by_bell(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _close_by(player, obj) * same_level(player, obj)


# ---------------------------------------------------------------------------------
# monkey logic
def close_by_monkey(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _close_by(player, obj) * same_level(player, obj)


# ---------------------------------------------------------------------------------
# coconut logic
def close_by_coconut_combi(player: th.Tensor, *objects: th.Tensor) -> th.Tensor:
    results = []
    for obj in objects:
        results.append(_close_by(player, obj) * same_level(player, obj))
    return th.stack(results).any(dim=0)


def close_by_coconut(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _close_by(player, obj) * same_level(player, obj)

# ---------------------------------------------------------------------------------
# general methods
def _close_by(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    th = 2  # Distance threshold
    player_x, player_y = player[:, 1], player[:, 2]
    obj_x, obj_y = obj[:, 1], obj[:, 2]
    obj_prob = obj[:, 0]
    dist = ((player_x - obj_x).pow(2) + (player_y - obj_y).pow(2)).sqrt()  # General distance
    return bool_to_probs(dist < th) * obj_prob * _in_field(obj)


def same_level(obj1: th.Tensor, obj2: th.Tensor) -> th.Tensor:
    obj1_y = obj1[..., 2]
    obj2_y = obj2[..., 2]

    is_3rd_level = th.logical_and(th.logical_and(28 < obj1_y, obj1_y < 76), th.logical_and(28 < obj2_y, obj2_y < 76))
    is_2nd_level = th.logical_and(th.logical_and(76 < obj1_y, obj1_y < 124), th.logical_and(76 < obj2_y, obj2_y < 124))
    is_1st_level = th.logical_and(th.logical_and(124 < obj1_y, obj1_y < 172),
                                  th.logical_and(124 < obj2_y, obj2_y < 172))

    is_same_level = th.logical_or(is_3rd_level, th.logical_or(is_2nd_level, is_1st_level))
    return bool_to_probs(is_same_level)


def is_lower(player: th.Tensor, *objects: th.Tensor, vertical_tolerance: float = 5) -> th.Tensor:
    """
    Check if one or more objects are horizontally on the same level or slightly lower than the player.
    """
    player_y = player[..., 2]
    results = []
    for obj in objects:
        obj_y, obj_prob = obj[..., 2], obj[:, 0]
        is_same_or_lower = (obj_y <= player_y) & (player_y - obj_y <= vertical_tolerance)
        results.append(bool_to_probs(is_same_or_lower) * obj_prob)
    return th.stack(results).any(dim=0)


def same_or_higher_level(player: th.Tensor, *objects: th.Tensor, vertical_tolerance: float = 5) -> th.Tensor:
    """
    Check if one or more objects are horizontally on the same level or slightly higher than the player.
    """
    player_y = player[..., 2]
    results = []
    for obj in objects:
        obj_y, obj_prob = obj[..., 2], obj[:, 0]
        is_same_or_higher = (obj_y >= player_y) & (obj_y - player_y <= vertical_tolerance)
        results.append(bool_to_probs(is_same_or_higher) * obj_prob)
    return th.stack(results).any(dim=0)


def _not_close_by(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_x = player[..., 1]
    player_y = player[..., 2]
    obj_x = obj[..., 1]
    obj_y = obj[..., 2]
    result = th.clip((abs(player_x - obj_x) + abs(player_y - obj_y) - 64) / 64, 0, 1)
    return result


def on_left(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """
    Check if the object is to the left of the player.
    """
    player_x = player[:, 1]
    obj_x = obj[:, 1]
    obj_prob = obj[:, 0]
    return bool_to_probs(obj_x < player_x) * obj_prob


def on_right(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """
    Check if the object is to the right of the player.
    """
    player_x = player[:, 1]
    obj_x = obj[:, 1]
    obj_prob = obj[:, 0]
    return bool_to_probs(obj_x > player_x) * obj_prob


def on_right_combi(player: th.Tensor, *objects: th.Tensor) -> th.Tensor:
    """
    Check if one or more objects are to the right of the player.
    """
    player_x = player[..., 1]
    results = []
    for obj in objects:
        obj_x, obj_prob = obj[..., 1], obj[:, 0]
        is_on_right = obj_x > player_x
        results.append(bool_to_probs(is_on_right) * obj_prob)
    return th.stack(results).any(dim=0)


def on_left_combi(player: th.Tensor, *objects: th.Tensor) -> th.Tensor:
    """
    Check if one or more objects are to the left of the player.
    """
    player_x = player[..., 1]
    results = []
    for obj in objects:
        obj_x, obj_prob = obj[..., 1], obj[:, 0]
        is_on_left = obj_x < player_x
        results.append(bool_to_probs(is_on_left) * obj_prob)
    return th.stack(results).any(dim=0)


def above(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    obj_prob = obj[:, 0]
    return bool_to_probs(obj_y > player_y) * obj_prob


def above_combi(player: th.Tensor, *objects: th.Tensor, vertical_tolerance: float = 60) -> th.Tensor:
    """
    Check if one or more objects are above the player.
    """
    player_y = player[..., 2]
    results = []
    for obj in objects:
        obj_y, obj_prob = obj[..., 2], obj[:, 0]
        is_above = obj_y > player_y
        results.append(bool_to_probs(is_above) * obj_prob)
    return th.stack(results).any(dim=0)


def not_close_by_missile(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _not_close_by(player, obj)


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
