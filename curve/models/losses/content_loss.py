import torch
import torch.nn as nn

from mmdet.models.registry import LOSSES
from mmdet.models.losses.utils import weighted_loss
from curve.core.bbox.transforms import reconstruct_cheby, reconstruct_fourier, f_series


@weighted_loss
def content_loss(pred, target, beta=1.0, sample_pts=360, encoding='cheby'):
    assert encoding in ['cheby', 'fourier', 'poly']
    assert beta > 0
    assert pred.size(0) == target.size(0) and target.numel() > 0, \
        "pred.shape:%s, target.shape:%s"%(pred.shape, target.shape)
    # pred size: N*26, target size: N*720
    num_coefs = pred.size(1)
    if encoding == 'cheby':
        theta = torch.linspace(-1, 1, sample_pts + 1)[:-1].cuda()
        fi = f_series(theta, num_coefs - 1)  # 0, ..., num_coefs-1 term
        pred_r = torch.mm(pred[:, :num_coefs], fi)
        gt_r = torch.mm(target[:, :num_coefs], fi)
        # pred_r = reconstruct_cheby(pred, sample_pts, num_coefs)
        # gt_r = reconstruct_cheby(target, sample_pts, num_coefs)
    elif encoding == 'fourier':
        pred_r = reconstruct_fourier(pred, sample_pts, num_coefs)
        gt_r = reconstruct_fourier(target, sample_pts, num_coefs)
    elif encoding == 'poly':
        raise NotImplementedError('Not Implemented Poly Reconstruction Loss.')

    diff = torch.abs(pred_r - gt_r)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    return loss


@LOSSES.register_module
class ContentLoss(nn.Module):

    def __init__(self, encoding='cheby', sample_pts=360, beta=1.0, reduction='mean', loss_weight=1.0):
        super(ContentLoss, self).__init__()
        self.encoding = encoding
        self.sample_pts = sample_pts
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight  # 0.2

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * content_loss(
            pred,
            target,
            weight,
            sample_pts=self.sample_pts,
            encoding=self.encoding,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox
