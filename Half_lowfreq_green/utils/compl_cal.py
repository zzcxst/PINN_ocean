import torch
import numpy as np

def compl_mul(x, y, is_torch=True):
    if is_torch:
        mul = torch.zeros_like(x)
    else:
        mul = np.zeros_like(x)
    mul[:, 0] = x[:, 0] * y[:, 0] - x[:, 1] * y[:, 1]
    mul[:, 1] = x[:, 0] * y[:, 1] + x[:, 1] * y[:, 0]
    return mul

def compl_div(x, y, is_torch=True):
    if is_torch:
        div = torch.zeros_like(x)
    else:
        div = np.zeros_like(x)
    div[:, 0] = (x[:, 0] * y[:, 0] + x[:, 1] * y[:, 1]) / (y[:, 0]**2 + y[:, 1]**2)
    div[:, 1] = (x[:, 1] * y[:, 0] - x[:, 0] * y[:, 1]) / (y[:, 0]**2 + y[:, 1]**2)
    return div
