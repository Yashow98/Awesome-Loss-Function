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
    
