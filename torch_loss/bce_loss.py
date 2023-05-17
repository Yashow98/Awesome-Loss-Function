# @Author  : YashowHoo
# @File    : bce_loss.py
# @Description : Implementation of the binary cross-entropy loss in numpy and torch

import numpy as np
import torch
from torch import nn


def np_bce_loss(y_pred, y_true):
    """
    Calculate the Binary Cross-Entropy loss in numpy
    :param y_pred:
    :param y_true:
    :return:
    """
    bce_loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()

    return bce_loss


def torch_bce_loss(y_pred, y_true):
    """
    Calculate the binary cross-entropy loss in torch
    :param y_pred:
    :param y_true:
    :return:
    """
    loss_func = nn.BCELoss()
    bce_loss = loss_func(y_pred, y_true)

    return bce_loss

if __name__ == '__main__':
    pred_array = np.array([0.1, 0.7, 0.9, 0.3])
    true_array = np.array([0, 1, 1, 0])

    bce_in_np = np_bce_loss(pred_array, true_array)
    print(bce_in_np)

    pred_tensor = torch.from_numpy(pred_array)
    true_tensor = torch.from_numpy(true_array)

    bce_in_torch = torch_bce_loss(pred_tensor, true_tensor.double())
    print(bce_in_torch.item())
