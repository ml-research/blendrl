import torch as th

from nsfr.utils.common import bool_to_probs


# def climbing(player: th.Tensor) -> th.Tensor:
#     status = player[..., 3]
#     return bool_to_probs(status == 12)


# def not_climbing(player: th.Tensor) -> th.Tensor:
#     status = player[..., 3
#     return bool_to_probs(status != 12)


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
    probs = th.stack(
        [
            _close_by(players[:, i, :], target_objs[:, i, :])
            for i in range(target_objs.size(1))
        ],
        dim=1,
    )

    max_closeby_prob, _ = probs.max(dim=1)
    result = (1.0 - max_closeby_prob).float()
    return result


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
    # dist = (player[:, 1:2] - obj[:, 1:2]).pow(2).sum(1).sqrt()
    return bool_to_probs(dist < th) * obj_prob * _in_field(obj)
    # result = th.clip((128 - abs(player_x - obj_x) - abs(player_y - obj_y)) / 128, 0, 1) * obj_prob
    # return result


def _not_close_by(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_x = player[..., 1]
    player_y = player[..., 2]
    obj_x = obj[..., 1]
    obj_y = obj[..., 2]
    result = th.clip((abs(player_x - obj_x) + abs(player_y - obj_y) - 64) / 64, 0, 1)
    return result


def not_close_by_missile(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    return _not_close_by(player, obj)


def not_close_by_enemy(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
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
