# @Author  : YashowHoo
# @File    : kld_loss.py
# @Description :KL Divergence loss

import numpy as np
import torch
from torch import nn


def np_kl_loss(y_pred, y_true):
    """
    Calculate kl loss

    :param y_pred:
    :param y_true:
    :return:
    """
    return np.sum(y_true * np.log(y_true / y_pred))


def torch_kl_loss(y_pred, y_true):
    """
    torch version

    :param y_pred:
    :param y_true:
    :return:
    """
    loss_func = nn.KLDivLoss(reduction="batchmean", log_target=False)
    return loss_func(y_pred, y_true)


if __name__ == '__main__':
    pred_array = np.array([0.4, 0.1, 0.7])
    true_array = np.array([0.1, 0.2, 0.9])

    kld_in_np = np_kl_loss(pred_array, true_array)
    print(kld_in_np)

    pred_tensor = torch.randn((3, 5))
    true_tensor = torch.randn((3, 5))

    kld_in_torch = torch_kl_loss(pred_tensor.softmax(dim=1).log(), true_tensor.softmax(dim=1))
    print(kld_in_torch.item())

