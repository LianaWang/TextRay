import torch
import numpy as np

from .geometry import iou_cuda
from shapely.geometry import Polygon


def poly_soft_nms_cpu(dets, sigma=0.5, min_score=0.01, decay='linear'):
    assert decay in ['linear', 'gaussian']
    dets = dets[dets[:, -1]>=0.5, :]
    boxes = dets[:, :-1]
    scores = dets[:, -1]
    
    # filter out non-clockwise polygons
#     valid = []
#     for i in range(boxes.shape[0]):
#         poly = Polygon(boxes[i, :].reshape((-1, 2)))
#         if poly.is_valid and poly.exterior.is_ccw:
#             valid.append(i)
#         else:
#             continue
#     boxes = boxes[valid, :]
#     scores = scores[valid]
#     dets = dets[valid, :]

    if len(boxes)<=0:
        return dets, []
    olp = iou_cuda(boxes.cuda()).cpu()
    keep = []
    N = boxes.shape[0]
    remained = list(range(N))

    decay_func = lambda x: 1.0 - x if decay == 'linear' else lambda x: np.exp(-(x * x) / sigma)

    while len(remained) > 0:
        remained_max = scores[remained].argmax()
        maxpos = remained[remained_max]
        keep.append(maxpos)
        remained.remove(maxpos)
        # update scores
        remained_list = remained[:]
        for pos in remained_list:
            iou = olp[maxpos, pos]
            if iou > 0:
                scores[pos] = decay_func(iou) * scores[pos]
                if scores[pos] <= min_score:
                    remained.remove(pos)
    dets = torch.cat([dets[keep, :-1], scores[keep, np.newaxis]], dim=1)
    return dets, keep


def bbox_olp(boxes):
    overlaps = np.zeros((boxes.shape[0], boxes.shape[0]))

    for k in range(len(boxes)):
        k_box = Polygon(boxes[k, :].reshape((-1, 2))).buffer(0)
        for n in range(k+1, len(boxes)):
            n_box = Polygon(boxes[n, :].reshape((-1, 2))).buffer(0)
            try:
                inter = k_box.intersection(n_box)
                area_union = k_box.area + n_box.area - inter.area
                overlaps[k,n] = inter.area / area_union
            except:
                overlaps[k,n] = 0
            overlaps[n, k] = overlaps[k,n]
        overlaps[k, k] = 1.0
    return overlaps


def poly_soft_nms(dets, sigma=0.5, min_score=1e-2, decay='linear'):
    """
    Soft NMS using polygon's iou
    :param dets: [k, d+1], final dimension stores detection scores
    :param sigma: used when decay is set to gaussian
    :param min_score: remove threshold of softened scores
    :param decay: decay method, can be linear or gaussian
    :return: kept detections and left indices
    """
    assert decay in ['linear', 'gaussian']
    if isinstance(dets, torch.Tensor):
        is_tensor = True
        dets = dets.detach() #.cpu().numpy()
    elif isinstance(dets, np.ndarray):
        is_tensor = False
        dets = torch.from_numpy(dets)
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    new_dets, inds = poly_soft_nms_cpu(
        dets,
        sigma=sigma,
        min_score=min_score,
        decay=decay)

    if is_tensor:
        return new_dets, torch.tensor(inds)
    else:
        return new_dets.astype(np.float32), inds.astype(np.int64)
