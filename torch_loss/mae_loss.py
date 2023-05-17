# @Author  : YashowHoo
# @File    : mae_loss.py
# @Description :mean absolute error loss or L1 loss

import numpy as np
from sklearn import metrics
import torch
from torch import nn


def np_mae_loss(y_pred, y_true):
    """
    Calculate mean absolute error loss in numpy
    :param y_pred:
    :param y_true:
    :return:
    """
    mae = np.abs(y_pred - y_true).mean()

    return mae


def sk_mae_loss(y_pred, y_true):
    """

    :param y_pred:
    :param y_true:
    :return:
    """
    return metrics.mean_absolute_error(y_true, y_pred)


def torch_mae_loss(y_pred, y_true):
    """

    :param y_pred:
    :param y_true:
    :return:
    """
    loss_func = nn.L1Loss()
    mae = loss_func(y_pred, y_true)

    return mae


if __name__ == '__main__':
    pred_array = np.array([0, 1, 2])
    true_array = np.array([0, 2, 4])

    mae_in_np = np_mae_loss(pred_array, true_array)
    print(mae_in_np)

    mae_in_sklearn = sk_mae_loss(pred_array, true_array)
    print(mae_in_sklearn)

    pred_tensor = torch.from_numpy(pred_array)
    true_tensor = torch.from_numpy(true_array)

    mae_in_torch = torch_mae_loss(pred_tensor.float(), true_tensor.float())
    print(mae_in_torch.item())

