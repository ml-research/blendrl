import torch as th

from nsfr.utils.common import bool_to_probs

def player_higher_than_ball(player: th.Tensor, ball: th.Tensor) -> th.Tensor:
    player_y = player[..., 1]
    ball_y = ball[..., 1]
    dist = player_y - ball_y
    beta = 1.0
    prob = th.sigmoid(dist / beta)
    return prob
    #return bool_to_probs(player_y - ball_y > 8)

def player_lower_than_ball(player: th.Tensor, ball: th.Tensor) -> th.Tensor:
    player_y = player[..., 1]
    ball_y = ball[..., 1]
    dist = player_y - ball_y
    beta = 1.0
    prob = th.sigmoid(-dist / beta)
    return prob
    #return bool_to_probs(player_y - ball_y < 8)

def ball_closeto_player(player: th.Tensor, ball: th.Tensor) -> th.Tensor:
    player_y = player[..., 1]
    ball_y = ball[..., 1]
    distance = abs(player_y - ball_y)
    return bool_to_probs(distance < 10)

def ball_goto_enemy(ball: th.Tensor) -> th.Tensor:
    ball_x = ball[...,0]
    ball_x2 = ball[...,2]
    return bool_to_probs(ball_x2 - ball_x > 0)

def ball_comingto_player(ball: th.Tensor) -> th.Tensor:
    ball_x = ball[...,0]
    ball_x2 = ball[...,2]
    return bool_to_probs(ball_x2 - ball_x < 0)








