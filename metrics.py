import numpy as np
import torch


def dice(mask1, mask2):
    A = mask2.view(-1)
    B = mask1.view(-1)

    intersection = (A * B).sum()
    union = A.sum() + B.sum()

    # Avoid zero division by adding small epsilon
    e = 1e-8
    dice_score = (2.0 * intersection + e) / (union + e)
    return dice_score.item()


def hd95(mask1, mask2):
    coords1 = torch.nonzero(mask1.squeeze(), as_tuple=False).numpy()
    coords2 = torch.nonzero(mask2.squeeze(), as_tuple=False).numpy()

    # If mask does not contain segmentation result (i.e. contains no pixel with value 1.)
    if (coords1.shape[0] == 0) or (coords2.shape[0] == 0):
        return np.inf

    # Pairwise distances
    dist = np.linalg.norm(coords1[:, None, :] - coords2[None, :, :], axis=-1)

    d_xy = np.min(dist, axis=1)
    d_yx = np.min(dist, axis=0)

    d_xy = np.percentile(d_xy, 95)
    d_yx = np.percentile(d_yx, 95)

    return max(d_xy, d_yx)
