# @Author  : YashowHoo
# @File    : huber_loss.py
# @Description :

import numpy as np
import torch
from torch import nn


def np_huber_loss(y_pred, y_true, delta=1.0):
    """
    Calculate huber loss

    :param y_pred:
    :param y_true:
    :return:
    """
    abs_error = np.abs(y_pred - y_true)
    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic

    return (0.5 * quadratic ** 2 + delta * linear).mean()


def torch_huber_loss(y_pred, y_true, delta=1.0):
    """
    torch version

    :param y_pred:
    :param y_true:
    :return:
    """
    loss_func = nn.HuberLoss(delta=delta)
    return loss_func(y_pred, y_true)


if __name__ == '__main__':
    pred_array = np.array([0.4, 0.1, 0.7])
    true_array = np.array([0.1, 0.2, 0.9])

    huber_in_np = np_huber_loss(pred_array, true_array, delta=1.3)
    print(huber_in_np)

    pred_tensor = torch.from_numpy(pred_array)
    ture_tensor = torch.from_numpy(true_array)

    huber_in_torch = torch_huber_loss(pred_tensor, ture_tensor, delta=1.3)
    print(huber_in_torch.item())

    print(torch.allclose(huber_in_torch, torch.tensor(huber_in_np)))

