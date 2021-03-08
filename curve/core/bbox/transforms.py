import math

import numpy as np
import numpy.polynomial.chebyshev as chebyshev
import torch


def bbox2offset(proposals, gt, means=0.0, stds=1.0):
    # proposals are anchors, deltas are offsets
    assert proposals.size(0) == gt.size(0)
    proposals = proposals.float()
    gt_box = gt.float()
    # print(gt_box)    
    px = (proposals[:, 0] + proposals[:, 2]) * 0.5
    py = (proposals[:, 1] + proposals[:, 3]) * 0.5
    pw = proposals[:, 2] - proposals[:, 0] + 1.0
    ph = proposals[:, 3] - proposals[:, 1] + 1.0
    
    gt_x_size = gt_box[:, 0::2].max(dim=1)[0] - gt_box[:, 0::2].min(dim=1)[0] + 1
    gt_y_size = gt_box[:, 1::2].max(dim=1)[0] - gt_box[:, 1::2].min(dim=1)[0] + 1
    gt_box_size = torch.max(gt_x_size, gt_y_size)

    deltas = torch.zeros((gt_box.size(0), gt_box.size(1)), dtype = torch.float).cuda()
    for i in range(int(gt_box.size(1)/2)):
        deltas[:, i*2] = torch.log(gt_box[:, i*2]+1) - torch.log(px+1)
        deltas[:, i*2+1] = torch.log(gt_box[:, i*2+1]+1) - torch.log(py+1)

    # centerness
    ctr_dist = gt_box[:, -2:] - torch.cat([px.unsqueeze(-1), py.unsqueeze(-1)], dim = -1)
    dist = torch.norm(ctr_dist, dim=1)
    weights = torch.clamp((gt_box_size - dist) / gt_box_size, min=1e-8, max=1.0)
    weights = weights/torch.mean(weights)
    weights = weights.detach()

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)
    return deltas, weights


def offset2bbox(rois, deltas, img_shape, scale_factor, means=0.0, stds=1.0):
    # rois are anchors, deltas are predicted offsets
    assert rois.size(0) == deltas.size(0)
    rois = rois.float()
    deltas = deltas.float()
    # centers
    px = ((rois[:, 0] + rois[:, 2]) * 0.5)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5)
    pw = (rois[:, 2] - rois[:, 0] + 1.0)
    ph = (rois[:, 3] - rois[:, 1] + 1.0)

    bboxes = torch.zeros((deltas.size(0), deltas.size(1)), dtype = torch.float)
    for i in range(int(deltas.size(1)/2)):
        bboxes[:, i*2] = (deltas[:, i*2] + torch.log(px+1)).exp()-1 #
        bboxes[:, i*2+1] = (deltas[:, i*2+1] + torch.log(py+1)).exp()-1 #
    bboxes = bboxes.cpu()
    bboxes = clip2img(bboxes, img_shape)
    bboxes /= scale_factor
    bboxes = bboxes[:, :-2]
    return bboxes

def bbox2radius(proposals, gt, skeleton, num_coords, means=0.0, stds=1.0):
    # proposals are anchors, deltas are offsets
    assert proposals.size(0) == gt.size(0)
    proposals = proposals.float()
    gt_box = gt.float()
    radiuses = skeleton
    rmax = skeleton.max(dim=1)[0]
    
    px = (proposals[:, 0] + proposals[:, 2]) * 0.5
    py = (proposals[:, 1] + proposals[:, 3]) * 0.5
    pw = proposals[:, 2] - proposals[:, 0] + 1.0
    ph = proposals[:, 3] - proposals[:, 1] + 1.0

    deltas = torch.zeros((gt_box.size(0), num_coords), dtype = torch.float).cuda()
    deltas[:, :-3] = radiuses / rmax.view(-1,1)
    deltas[:, -3] = (rmax - 180.0) / 65.0
    deltas[:, -2] = torch.log(gt_box[:, -2] + 1) - torch.log(px + 1)
    deltas[:, -1] = torch.log(gt_box[:, -1] + 1) - torch.log(py + 1)
    
    # centerness
    ctr_dist = gt_box[:, -2:] - torch.cat([px.unsqueeze(-1), py.unsqueeze(-1)], dim = -1)
    dist = torch.norm(ctr_dist, dim=1)
    weights = torch.clamp((rmax - dist) / rmax, min=1e-8, max=1.0)
    weights = weights/torch.mean(weights)
    weights = weights.detach()

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)
    
    return deltas, weights


def radius2bbox(rois, deltas, img_shape, scale_factor, means=0.0, stds=1.0,):
    # rois are anchors, deltas are predicted offsets
    assert rois.size(0) == deltas.size(0)
    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas * stds + means

    rois = rois.float()
    deltas = deltas.float()
    px = ((rois[:, 0] + rois[:, 2]) * 0.5)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5)

    rmax = deltas[:, -3] * 65.0 + 180.0
    r = deltas[:, :-3]
    center_x = (deltas[:, -2] + torch.log(px + 1)).exp() - 1
    center_y = (deltas[:, -1] + torch.log(py + 1)).exp() - 1

    contours = torch.zeros((deltas.shape[0], 360 * 2))
    theta = torch.linspace(-1, 1, 361)[:-1].cuda()
    contours[:, 0::2] = r * torch.cos(theta * np.pi)[None, :] * rmax[:, None] + center_x[:, None]
    contours[:, 1::2] = r * torch.sin(theta * np.pi)[None, :] * rmax[:, None] + center_y[:, None]
    
    contours = get_uniform_points(contours).cpu()
    contours = clip2img(contours, img_shape)
    contours /= scale_factor
    return contours


def bbox2cheby(proposals, pred, gt, skeleton, num_coords, means=0.0, stds=1.0):
    """
        - proposals: anchors
        - pred: 26d, = [23] + [3]
        - gt: 26d
    """
    # proposals are anchors, deltas are offsets
    assert proposals.size(0) == gt.size(0)
    proposals = proposals.float()
    cheby = gt.float()
    px = (proposals[:, 0] + proposals[:, 2]) * 0.5
    py = (proposals[:, 1] + proposals[:, 3]) * 0.5
    pw = proposals[:, 2] - proposals[:, 0] + 1.0
    ph = proposals[:, 3] - proposals[:, 1] + 1.0
    deltas = cheby.new_zeros((cheby.shape[0], num_coords))
    deltas[:, :num_coords-3] = cheby[:, :num_coords-3]
    deltas[:, -3] = (cheby[:, -3] - 180.0) / 65.0
    deltas[:, -2] = torch.log(cheby[:, -2] + 1) - torch.log(px + 1)
    deltas[:, -1] = torch.log(cheby[:, -1] + 1) - torch.log(py + 1)

    # centerness
    ctr_dist = cheby[:, -2:] - torch.cat([px.unsqueeze(-1), py.unsqueeze(-1)], dim = -1)
    dist = torch.norm(ctr_dist, dim=1)
    weights = torch.clamp((cheby[:, -3] - dist) / cheby[:, -3], min=1e-8, max=1.0)
    weights = weights/torch.mean(weights)
    weights = weights.detach()

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)
    return deltas, weights


def bbox2fori(proposals, pred, gt, skeleton, means=0.0, stds=1.0, num_coords=26):

    # proposals are anchors, deltas are offsets
    num_coefs = num_coords - 3
    assert proposals.size(0) == gt.size(0)

    proposals = proposals.float()
    fori = gt.float()
    px = (proposals[:, 0] + proposals[:, 2]) * 0.5
    py = (proposals[:, 1] + proposals[:, 3]) * 0.5
    pw = proposals[:, 2] - proposals[:, 0] + 1.0
    ph = proposals[:, 3] - proposals[:, 1] + 1.0
    deltas = fori.new_zeros((fori.shape[0], num_coords))
    deltas[:, :num_coefs] = fori[:, :num_coefs]
    deltas[:, num_coefs] = (fori[:, num_coefs] - 180.0) / 65.0
    deltas[:, num_coefs + 1] = torch.log(fori[:, num_coefs + 1] + 1) - torch.log(px + 1)
    deltas[:, num_coefs + 2] = torch.log(fori[:, num_coefs + 2] + 1) - torch.log(py + 1)

    # centerness
    ctr_dist = fori[:, -2:] - torch.cat([px.unsqueeze(-1), py.unsqueeze(-1)], dim=-1)
    dist = torch.norm(ctr_dist, dim=1)
    weights = torch.clamp((fori[:, -3] - dist) / fori[:, -3], min=1e-8, max=1.0)
    weights = weights / torch.mean(weights)
    weights = weights.detach()

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)
    return deltas, weights

fi = None
def reconstruct_cheby(coefs, sample_pts=360, num_coefs=23):
    global fi
    if fi is None:
        theta = torch.linspace(-1, 1, sample_pts + 1)[:-1].cuda()
        fi = f_series(theta, num_coefs - 1)  # 0, ..., num_coefs-1 term
    # ((N*23 dot 23*360).T * N).T----->N*36
    r = torch.mm(coefs[:, :num_coefs], fi)
#     xy = coefs.new_zeros((coefs.shape[0], 36*2))
#     xy[:, 0::2] = r * torch.cos(theta * np.pi) #* coefs[:, 23].unsqueeze(1) + coefs[:, 24].unsqueeze(1)
#     xy[:, 1::2] = r * torch.sin(theta * np.pi) #* coefs[:, 23].unsqueeze(1) + coefs[:, 25].unsqueeze(1)
    return r

def f_series(x, n):
    y = x.repeat(n+1, 1)
    # x---360; y---23*360
    y[0, :] = 1
    for i in range(2, n+1):
        y[i, :] = 2.0 * x * y[i-1, :] - y[i-2, :]
    return y

def get_uniform_points(ad_points):
    points = torch.roll(ad_points.reshape(-1, 360, 2), shifts=-5, dims=1)
    dists = (torch.roll(points, shifts=-1, dims=1) - points).norm(dim=-1).cumsum(dim=-1).cpu().numpy()
    uni_points = torch.zeros((points.size(0), 36, 2))
    for i in range(len(dists)):
        inds = np.searchsorted(dists[i], np.histogram(dists[i], bins=uni_points.size(1))[1])[:-1]
        uni_points[i, :, :] = points[i, inds, :]
    return uni_points.view(-1, 72)

def get_uniform_points_fast(ad_points):
    points = torch.mean(ad_points.reshape(-1, 36, 10, 2), dim=2)
    return points.view(-1, 72)

theta_cos = None
theta_sin = None
def cheby2bbox(rois, deltas, img_shape, scale_factor, num_coords, means=0.0, stds=1.0,
               sample_pts=36):
    # rois are anchors, deltas are predicted offsets

    duplicates = 10
    assert rois.size(0) == deltas.size(0)
    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas * stds + means

    
    rois = rois.float()
    deltas = deltas.float()
    px = ((rois[:, 0] + rois[:, 2]) * 0.5)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5)

    cheby = deltas[:, :-3]
    rmax = deltas[:, -3] * 65.0 + 180.0
    center_x = (deltas[:, -2] + torch.log(px + 1)).exp() - 1
    center_y = (deltas[:, -1] + torch.log(py + 1)).exp() - 1
    r = reconstruct_cheby(cheby, sample_pts * duplicates, num_coords-3)
    
    global theta_cos, theta_sin
    if theta_cos is None or theta_sin is None:
        theta = torch.linspace(-1, 1, sample_pts * duplicates + 1)[:-1].cuda()
        theta_cos = torch.cos(theta * np.pi)[None, :]
        theta_sin = torch.sin(theta * np.pi)[None, :]
    contours = deltas.new_zeros((deltas.shape[0], sample_pts * duplicates * 2))
    contours[:, 0::2] = r * theta_cos * rmax[:, None] + center_x[:, None]
    contours[:, 1::2] = r * theta_sin * rmax[:, None] + center_y[:, None]

    contours = get_uniform_points(contours).cpu()        # smooth but slow
    # contours = get_uniform_points_fast(contours).cpu() # fast but bumpy

    contours = clip2img(contours, img_shape)
    contours /= scale_factor
    return contours


def fori2bbox(rois, deltas, img_shape, scale_factor, means=[0, 0, 0, 0], stds=[1, 1, 1, 1],
              sample_pts=36, num_coords=51):
    num_coefs = num_coords - 3
    duplicates = 10

    # rois are anchors, deltas are predicted offsets
    assert rois.size(0) == deltas.size(0)
    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas * stds + means  # those are predicated fourier coefs

    rois = rois.float()
    deltas = deltas.float()
    px = ((rois[:, 0] + rois[:, 2]) * 0.5)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5)
    r = reconstruct_fourier(deltas[:, :num_coefs], sample_pts * duplicates, num_coefs)  # [N, sample_pts * dupliactes]
    r = r[:, :sample_pts * duplicates]  # take the real part

    r = r.reshape((-1, sample_pts, duplicates)).mean(dim=-1)  # [N, sample_pts]

    rmax = deltas[:, num_coefs] * 65.0 + 180.0
    center_x = (deltas[:, num_coefs + 1] + torch.log(px + 1)).exp() - 1
    center_y = (deltas[:, num_coefs + 2] + torch.log(py + 1)).exp() - 1

    contours = torch.zeros((deltas.shape[0], sample_pts * 2))
    theta = torch.linspace(-1, 1, sample_pts * duplicates + 1)[:-1].cuda()
    theta = theta[duplicates // 2::duplicates] # take the average degrees

    contours[:, 0::2] = r * torch.cos(theta * np.pi)[None, :] * rmax[:, None] + center_x[:, None]
    contours[:, 1::2] = r * torch.sin(theta * np.pi)[None, :] * rmax[:, None] + center_y[:, None]
    contours = contours.cpu()
    contours = get_uniform_points(contours).cpu()
    contours = clip2img(contours, img_shape)
    contours /= scale_factor
    return contours


def reconstruct_fourier(foris, sample_pts, num_coefs):
    """
    :param foris: [N, num_coefs]
    :param sample_pts:
    :param num_coefs:
    :return: [N, sample_pts * 2]
    """
    theta = torch.arange(sample_pts).cuda(foris.device) * np.pi * 2 / sample_pts  # [sample_points]
    i = torch.arange(num_coefs // 2).cuda(foris.device)  # num_coefs contains two parts
    itheta = i[:, None] * theta[None, :]

    foris_real = foris[:, :num_coefs // 2]
    foris_imag = foris[:, num_coefs // 2:num_coefs]

    base_comps_real = torch.cos(itheta)
    base_comps_imag = torch.sin(itheta)
    # [N, num_coefs // 2] dot [num_coefs //2, sample_pts]
    r_real = torch.mm(foris_real, base_comps_real) - torch.mm(foris_imag, base_comps_imag)
    r_real = r_real / (num_coefs // 2)
    r_imag = torch.mm(foris_real, base_comps_imag) + torch.mm(foris_imag, base_comps_real)
    r_imag = r_imag / (num_coefs // 2)
    return torch.cat([r_real, r_imag], dim=-1)


def clip2img(contours, img_shape):
    # 0 <= x <= im_shape[1]
    contours[:, 0::2] = np.maximum(np.minimum(contours[:, 0::2], img_shape[1] - 1), 0)
    # 0 <= y < im_shape[0]
    contours[:, 1::2] = np.maximum(np.minimum(contours[:, 1::2], img_shape[0] - 1), 0)
    return contours
