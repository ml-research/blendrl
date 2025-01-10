import torch as th

from nsfr.utils.common import bool_to_probs


def player_higher_than_ball(player: th.Tensor, ball: th.Tensor) -> th.Tensor:
    player_y = player[..., 2]
    ball_y = ball[..., 2]
    is_higher = player_y > ball_y
    return bool_to_probs(is_higher)


def player_lower_than_ball(player: th.Tensor, ball: th.Tensor) -> th.Tensor:
    player_y = player[..., 2]
    ball_y = ball[..., 2]
    is_lower = player_y < ball_y
    return bool_to_probs(is_lower)


def ball_close_to_player(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    th = 5
    player_x, player_y = player[:, 1], player[:, 2]
    obj_x, obj_y = obj[:, 1], obj[:, 2]
    obj_prob = obj[:, 0]
    dist = ((player_x - obj_x).pow(2) + (player_y - obj_y).pow(2)).sqrt()
    return bool_to_probs(dist < th) * obj_prob


def ball_close_to_enemy(enemy: th.Tensor, obj: th.Tensor) -> th.Tensor:
    th = 2
    enemy_x, enemy_y = enemy[:, 1], enemy[:, 2]
    obj_x, obj_y = obj[:, 1], obj[:, 2]
    obj_prob = obj[:, 0]
    dist = ((enemy_x - obj_x).pow(2) + (enemy_y - obj_y).pow(2)).sqrt()
    return bool_to_probs(dist < th) * obj_prob


def ball_is_on_left(player: th.Tensor, ball: th.Tensor, y_tolerance: float = 10.0) -> th.Tensor:
    """
    Determines if the ball is on the left of the player based on both x and y coordinates.
    """
    ball_x = ball[..., 1]
    player_x = player[..., 1]
    ball_y = ball[..., 2]
    player_y = player[..., 2]
    is_left = ball_x < player_x
    is_aligned_y = th.abs(player_y - ball_y) <= y_tolerance
    is_on_left = th.logical_and(is_left, is_aligned_y)
    return bool_to_probs(is_on_left)


def ball_is_on_right(player: th.Tensor, ball: th.Tensor, y_tolerance: float = 10.0) -> th.Tensor:
    """
    Determines if the ball is on the right of the player based on both x and y coordinates.
    """
    ball_x = ball[..., 1]
    player_x = player[..., 1]
    ball_y = ball[..., 2]
    player_y = player[..., 2]
    is_right = ball_x > player_x
    is_aligned_y = th.abs(player_y - ball_y) <= y_tolerance
    is_on_right = th.logical_and(is_right, is_aligned_y)
    return bool_to_probs(is_on_right)