import numpy as np
import torch
from shapely.geometry import Polygon
from torch.nn.functional import interpolate


def image_from_tensor(img, img_meta):
    h, w, _ = img_meta[0]['img_shape']
    img = img[:, :, :h, :w]
    h, w, _ = img_meta[0]['ori_shape']
    img = interpolate(img, (h, w))
    img = denormalize(img[0])
    return img


def denormalize(img, mean=np.array([[[123.675, 116.28, 103.53]]]),
                var=np.array([[[58.395, 57.12, 57.375]]]),
                ):
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    if isinstance(mean, (tuple, list)):
        mean = np.array(mean).reshape([1, 1, 3])
    if isinstance(var, (tuple, list)):
        var = np.array(var).reshape([1, 1, 3])

    img = np.transpose(img, (1, 2, 0))
    img *= var
    img += mean
    img = img.astype(np.uint8)
    return img


def is_clockwise(a):
    p = Polygon(a)
    if not p.is_valid:
        print("not valid")
        return False
    x = a[:, 0]
    y = a[:, 1]
    area = 0.5 * (np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area < 0


def plot_cheby_bboxes(bboxes, ax):
    assert bboxes.shape[1] == 72
    bboxes = bboxes.reshape((-1, 36, 2))
    for bbox in bboxes:
        ax.plot(bbox[:36, 0], bbox[:36, 1], color='red', linewidth=3)
        ax.plot(bbox[0, 0], bbox[0, 1], 'x', color='green', markeredgewidth=2, markersize=5)
        ax.plot(bbox[-1, 0], bbox[-1, 1], 'x', color='orange', markeredgewidth=2, markersize=5)


def plot_offset_bboxes(bboxes, ax):
    assert bboxes.shape[1] == 24
    bboxes = bboxes.reshape((-1, 12, 2))
    for bbox in bboxes:
        ax.plot(bbox[:12, 0], bbox[:12, 1], color='red')
        ax.plot(bbox[:, 0], bbox[:, 1], 'x', color='red', markeredgewidth=2, markersize=5)
