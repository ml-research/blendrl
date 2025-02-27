import torch as th


def player_higher_than_ball(player: th.Tensor, ball: th.Tensor) -> th.Tensor:
    player_y = player[..., 1]
    ball_y = ball[..., 1]
    dist = player_y - ball_y
    return sigmoid_smoothing(dist > 8, temperature=4.0)
    # return bool_to_probs(player_y - ball_y > 8)


def player_lower_than_ball(player: th.Tensor, ball: th.Tensor) -> th.Tensor:
    player_y = player[..., 1]
    ball_y = ball[..., 1]
    dist = player_y - ball_y
    return sigmoid_smoothing(dist < 8, temperature=4.0)


def ball_closeto_player(player: th.Tensor, ball: th.Tensor) -> th.Tensor:
    player_y = player[..., 1]
    ball_y = ball[..., 1]
    distance = abs(player_y - ball_y)
    return sigmoid_smoothing(distance < 10, temperature=6.0)


def ball_goto_enemy(ball: th.Tensor) -> th.Tensor:
    ball_x = ball[..., 0]
    ball_x2 = ball[..., 2]
    return sigmoid_smoothing(ball_x2 - ball_x > 0,temperature=5.0)


def ball_comingto_player(ball: th.Tensor) -> th.Tensor:
    ball_x = ball[..., 0]
    ball_x2 = ball[..., 2]
    return sigmoid_smoothing(ball_x2 - ball_x < 0, temperature=5.0)

def sigmoid_smoothing(bool_tensor: th.Tensor, temperature: float = 5.0) -> th.Tensor:
    """
    Apply sigmoid smoothing to a boolean tensor, converting True/False into soft probabilities.

    :param bool_tensor: Boolean tensor indicating condition (True = overlap, False = no overlap).
    :param temperature: Controls softness of probability conversion (higher = more binary-like).
    :return: Soft probability tensor (0.0 to 1.0).
    """
    return th.sigmoid(temperature * (bool_tensor.float() - 0.5))  # Adaptive smoothing