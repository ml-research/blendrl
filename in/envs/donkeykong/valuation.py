import torch as th

from nsfr.utils.common import bool_to_probs


# def climbing(player: th.Tensor) -> th.Tensor:
#     status = player[..., 3]
#     return bool_to_probs(status == 12)


# def not_climbing(player: th.Tensor) -> th.Tensor:
#     status = player[..., 3
#     return bool_to_probs(status != 12)

def _in_field(obj: th.Tensor) -> th.Tensor:
    x = obj[..., 1]
    prob = obj[:, 0]
    return bool_to_probs(th.logical_and(16 <= x, x <= 16 + 128)) * prob


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
    x_prob = bool_to_probs(abs(player_x - obj_x) < 2)
    return x_prob
    # y_prob = bool_to_probs(obj_y > player_y - 8)
    # return  x_prob * y_prob * obj_prob * same_level_ladder(player, obj)


def same_level_ladder(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    obj_prob = obj[:, 0]
    return bool_to_probs(abs(player_y - obj_y) < 30) * obj_prob


def same_level_ladder(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    obj1_y = player[..., 2] + 8
    obj2_y = obj[..., 2]
    z = 5  # floor thickness
    # eps = 12

    is_1st_level = th.logical_and(th.logical_and(176 - 5 < obj1_y, obj1_y < 193 + 8),
                                  th.logical_and(176 < obj2_y, obj2_y < 193))
    is_2nd_level = th.logical_and(th.logical_and(148 - 5 < obj1_y, obj1_y < 165 + 8),
                                  th.logical_and(148 < obj2_y, obj2_y < 165))
    is_3rd_level = th.logical_and(th.logical_and(120 - 5 < obj1_y, obj1_y < 137 + 8),
                                  th.logical_and(120 < obj2_y, obj2_y < 137))
    is_4th_level = th.logical_and(th.logical_and(92 - 5 < obj1_y, obj1_y < 109 + 8),
                                  th.logical_and(92 < obj2_y, obj2_y < 109))
    is_5th_level = th.logical_and(th.logical_and(64 - 5 < obj1_y, obj1_y < 81 + 8),
                                  th.logical_and(64 < obj2_y, obj2_y < 81))
    is_6th_level = th.logical_and(th.logical_and(40 - 5 < obj1_y, obj1_y < 57 + 8),
                                  th.logical_and(40 < obj2_y, obj2_y < 57))

    is_same_level_1 = th.logical_or(is_3rd_level, th.logical_or(is_2nd_level, is_1st_level))
    is_same_level_2 = th.logical_or(is_4th_level, th.logical_or(is_5th_level, is_6th_level))
    is_same_level = th.logical_or(is_same_level_1, is_same_level_2)
    return bool_to_probs(is_same_level)


def same_level(obj1: th.Tensor, obj2: th.Tensor) -> th.Tensor:
    obj1_y = obj1[..., 2]
    obj2_y = obj2[..., 2]

    is_3rd_level = th.logical_and(th.logical_and(28 < obj1_y, obj1_y < 76), th.logical_and(28 < obj2_y, obj2_y < 76))
    is_2nd_level = th.logical_and(th.logical_and(76 < obj1_y, obj1_y < 124), th.logical_and(76 < obj2_y, obj2_y < 124))
    is_1st_level = th.logical_and(th.logical_and(124 < obj1_y, obj1_y < 172),
                                  th.logical_and(124 < obj2_y, obj2_y < 172))

    is_same_level = th.logical_or(is_3rd_level, th.logical_or(is_2nd_level, is_1st_level))
    return bool_to_probs(is_same_level)


def obj_above_ladder(obj: th.Tensor, ladder: th.Tensor) -> th.Tensor:
    """
    Check if the object is above the ladder.
    """
    barrel_x, barrel_y = obj[..., 1], obj[..., 2]
    ladder_x, ladder_y = ladder[..., 1], ladder[..., 2]
    barrel_prob = obj[:, 0]
    is_above_ladder = (barrel_y < ladder_y) & (ladder_y - barrel_y < 60)
    is_aligned_with_ladder = abs(barrel_x - ladder_x) < 5
    return bool_to_probs(is_above_ladder & is_aligned_with_ladder) * barrel_prob


def safe_to_climb(*objects: th.Tensor, ladder: th.Tensor, platform: th.Tensor) -> th.Tensor:
    """
    Check if it is safe to climb the ladder and the barrel is not directly on the platform.
    """
    results = []
    for obj in objects:
        hazard_on_platform = obj_above_ladder(obj, ladder) & _on_platform(obj, platform)
        results.append(~bool_to_probs(hazard_on_platform))
    return th.stack(results).any(dim=0)


def _close_by(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    th = 32
    player_x = player[:, 1]
    player_y = player[:, 2]
    obj_x = obj[:, 1]
    obj_y = obj[:, 2]
    obj_prob = obj[:, 0]
    x_dist = (player_x - obj_x).pow(2)
    y_dist = (player_y - obj_y).pow(2)
    dist = (x_dist + y_dist).sqrt()
    return bool_to_probs(dist < th) * obj_prob * _in_field(obj)


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
    return bool_to_probs(obj_x < player_x) * obj_prob * same_level_ladder(player, obj)


def on_right(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """
    Check if the object is to the right of the player.
    """
    player_x = player[:, 1]
    obj_x = obj[:, 1]
    obj_prob = obj[:, 0]
    return bool_to_probs(obj_x > player_x) * obj_prob * same_level_ladder(player, obj)


def above(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    obj_prob = obj[:, 0]
    return bool_to_probs(obj_y > player_y) * obj_prob


def close_by_barrel(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _close_by(player, obj) * same_level(player, obj)


def close_by_hammer(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _close_by(player, obj) * same_level(player, obj)


def has_hammer(player: th.Tensor, hammer: th.Tensor) -> th.Tensor:
    """
    Dynamically check if the player has collected the hammer based on proximity.
    """
    # Extract coordinates
    player_x, player_y = player[..., 1], player[..., 2]
    hammer_x, hammer_y = hammer[..., 1], hammer[..., 2]
    hammer_prob = hammer[..., 0]
    is_close = th.logical_and(
        th.abs(player_x - hammer_x) < 5,  # Within 5 units horizontally
        th.abs(player_y - hammer_y) < 5   # Within 5 units vertically
    )
    # Determine collection probability based on proximity and hammer presence
    collected_prob = bool_to_probs(is_close) * hammer_prob
    # A hammer is considered collected if the probability exceeds a threshold (e.g., >0.5)
    is_collected = collected_prob > 0.5
    return is_collected


def has_no_hammer(player: th.Tensor, hammers: th.Tensor) -> th.Tensor:
    """
    Check if the player has no hammer.
    """
    player_x, player_y = player[..., 1], player[..., 2]
    no_hammer = th.ones_like(player[..., 0], dtype=th.bool)
    for hammer in hammers:
        hammer_x, hammer_y = hammer[..., 1], hammer[..., 2]
        hammer_prob = hammer[..., 0]
        is_close = th.logical_and(
            th.abs(player_x - hammer_x) < 5,  # Within 5 units horizontally
            th.abs(player_y - hammer_y) < 5   # Within 5 units vertically
        )
        has_hammer = bool_to_probs(is_close) * hammer_prob > 0.5
        no_hammer = th.logical_and(no_hammer, ~has_hammer)
    return bool_to_probs(no_hammer)


def no_donkeykong_between(player: th.Tensor, girlfriend: th.Tensor, donkeykong: th.Tensor) -> th.Tensor:
    """
    Check if there is no Donkey Kong between the player and the girlfriend.
    """
    player_x, player_y = player[..., 1], player[..., 2]
    girlfriend_x, girlfriend_y = girlfriend[..., 1], girlfriend[..., 2]
    donkeykong_x, donkeykong_y = donkeykong[..., 1], donkeykong[..., 2]
    is_between_x = th.logical_and(
        th.min(player_x, girlfriend_x) < donkeykong_x,
        donkeykong_x < th.max(player_x, girlfriend_x)
    )
    same_level = th.abs(player_y - donkeykong_y) < 10
    no_dk_between = ~th.logical_and(is_between_x, same_level)

    # Return the probability of no Donkey Kong being between (adjusted for DK's presence probability)
    return bool_to_probs(no_dk_between) * (1 - donkeykong[:, 0])


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
