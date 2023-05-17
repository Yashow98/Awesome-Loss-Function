# @Author  : Yashowhoo
# @File    : mse_loss.py
# @Description :Implementation of mse loss in numpy, sklearn and torch.

import numpy as np
import torch
from torch import nn
from sklearn import metrics

def np_mse_loss(y_pred, y_true):
    """
    Calculate the mean squared error (MSE) loss between predicted and true values.

    :param y_pred: predicted values, are numpy arrays
    :param y_true: true values, are numpy arrays
    :return: the mean squared error loss
    """
    n = len(y_true)
    mse_loss = np.sum((y_pred - y_true) ** 2) / n

    return mse_loss


def sklearn_mse_loss(y_pred, y_true):
    """

    :param y_pred:
    :param y_true:
    :return:
    """
    return metrics.mean_squared_error(y_true, y_pred)


def torch_mse_loss(y_pred, y_true):
    """
    mse loss in torch

    :param y_pred: predicted values, are tensors
    :param y_true: true values, are tensors
    :return:mse loss
    """
    loss_func = nn.MSELoss()
    mse_loss = loss_func(y_pred, y_true)

    return mse_loss

if __name__ == '__main__':
    pred_array = np.array([0, 1, 2])
    true_array = np.array([0, 2, 4])

    np_mse = np_mse_loss(pred_array, true_array)
    print(np_mse)

    sk_mse = sklearn_mse_loss(pred_array, true_array)
    print(sk_mse)

    pred_tensor = torch.from_numpy(pred_array)
    true_tensor = torch.from_numpy(true_array)

    torch_mse = torch_mse_loss(pred_tensor.float(), true_tensor.float())
    print(torch_mse.item())
