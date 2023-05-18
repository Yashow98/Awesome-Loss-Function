# @Author  : YashowHoo
# @File    : hinge_loss.py
# @Description :
import numpy as np
import torch
from torch import nn
from sklearn import metrics


def np_hinge_loss(y_pred, y_true, margin=1.0):
    """
    Calculate hinge loss in numpy

    :param y_pred:
    :param y_true:
    :return:
    """
    hinge = np.maximum(0, margin - (y_pred * y_true)).mean()

    return hinge


def torch_hinge_loss(y_pred, y_true, margin=1.0):
    """
    torch version

    :param y_pred:
    :param y_true:
    :param margin:
    :return:
    """
    loss_func = nn.HingeEmbeddingLoss(margin=margin)
    return loss_func(y_pred, y_true)


if __name__ == '__main__':
    pred_arr = np.array([0.1, 1.5, 0.9, 1.2])
    true_arr = np.array([-1, 1., 1, 1])

    hinge_in_np = np_hinge_loss(pred_arr, true_arr)
    print(hinge_in_np)

    hinge_in_sk = metrics.hinge_loss(true_arr, pred_arr)
    print(hinge_in_sk)

    pred_tensor = torch.from_numpy(pred_arr)
    true_tensor = torch.from_numpy(true_arr)

    hinge_in_torch = torch_hinge_loss(pred_tensor, true_tensor)
    print(hinge_in_torch.item())

