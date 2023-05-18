# @Author  : YashowHoo
# @File    : dice_loss.py
# @Description : Dice loss, also known as the Sorensen-Dice coefficient or F1 score, is a loss
# function used in image segmentation tasks to measure the overlap between
# the predicted segmentation and the ground truth. The Dice ranges from
# 0 to 1, where 0 indicates no overlap and 1 indicates perfect overlap.

import numpy as np
import torch
from torch import nn
from sklearn import metrics


def np_dice_loss(y_pred, y_true, smooth=1e-05):
    """
    Calculate dice loss in numpy

    :param y_pred:
    :param y_true:
    :param smooth:
    :return:
    """
    intersection = np.sum(y_pred * y_true)
    sum_of_squares_pred = np.sum(np.square(y_pred))
    sum_of_squares_true = np.sum(np.square(y_true))
    dice = 1 - (2 * intersection + smooth) / (sum_of_squares_pred + sum_of_squares_true)

    return dice


def torch_dice_loss(y_pred, y_true, smooth=1e-05):
    """
    torch version

    :param y_pred:
    :param y_true:
    :return:
    """
    intersection = torch.sum(y_pred * y_true)
    sum_of_squares_pred = torch.sum(torch.square(y_pred))
    sum_of_squares_true = torch.sum(torch.square(y_true))
    dice = 1 - (2 * intersection + smooth) / (sum_of_squares_pred + sum_of_squares_true)

    return dice


if __name__ == '__main__':
    pred_arr = np.array([1, 1, 0, 1, 1, 0])
    true_arr = np.array([1, 1, 0, 1, 0, 1])

    dice_in_np = np_dice_loss(pred_arr, true_arr)
    print(dice_in_np)

    pred_tensor = torch.from_numpy(pred_arr)
    true_tensor = torch.from_numpy(true_arr)

    dice_in_torch = torch_dice_loss(pred_tensor, true_tensor)
    print(dice_in_torch.item())
