# @Author  : YashowHoo
# @File    : cce_loss.py
# @Description :categorical cross-entropy loss

import numpy as np
import torch
from torch import nn
from sklearn import metrics


def np_cce_loss(y_pred, y_true):
    """
    Calculate cce loss in numpy

    :param y_pred:
    :param y_true: one-hot encode
    :return:
    """
    cce_loss = -(np.sum(y_true * np.log(y_pred))).mean()

    return cce_loss


def torch_cce_loss(y_pred, y_true):
    """
    torch version

    :param y_pred: input logits
    :param y_true: [num_class]
    :return:
    """
    loss_func = nn.CrossEntropyLoss()

    return loss_func(y_pred, y_true)


if __name__ == '__main__':
    pred_array = np.array([[0.8, 0.1, 0.1],
                           [0.7, 0.2, 0.1],
                           [0.1, 0.3, 0.6]])
    # one-hot encode
    true_array = np.array([[1, 0, 0],
                           [1, 0, 0],
                           [0, 0, 1]])
    cce_in_np = np_cce_loss(pred_array, true_array)
    print(cce_in_np)

    pred_tensor = torch.from_numpy(pred_array)
    true_tensor = torch.tensor([0, 0, 2])

    cce_in_torch = torch_cce_loss(pred_tensor, true_tensor)
    print(cce_in_torch.item())
