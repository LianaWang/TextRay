import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.models.registry import HEADS
from .anchor_head import ChebyAnchorHead, OffsetAnchorHead, CurveAnchorHead, FourierAnchorHead, RadiusAnchorHead
from curve.core import offset2bbox, cheby2bbox, fori2bbox, radius2bbox
from curve.ops import poly_soft_nms
from curve.models.plugins import ConvDU, ConvLR
from mmdet.ops import DeformConvPack


class CurveRPNHead(CurveAnchorHead):
    def __init__(self, in_channels, du_cfg=None, lr_cfg=None, dfm_cfg=None, apply_dulr=False, **kwargs):
        if du_cfg is None and apply_dulr:  # for compatibility
            du_cfg = dict(in_out_channels=in_channels, kernel_size=(1, 9), groups=1)
        if lr_cfg is None and apply_dulr:
            lr_cfg = dict(in_out_channels=in_channels, kernel_size=(9, 1), groups=1)

        self.du_cfg = du_cfg
        self.lr_cfg = lr_cfg
        self.dfm_cfg = dfm_cfg
        super(CurveRPNHead, self).__init__(num_classes=2, in_channels=in_channels, **kwargs)

    @property
    def with_dfm(self):
        return hasattr(self, 'dfm_cfg') and self.dfm_cfg is not None

    @property
    def with_du(self):
        return hasattr(self, 'du_cfg') and self.du_cfg is not None

    @property
    def with_lr(self):
        return hasattr(self, 'lr_cfg') and self.lr_cfg is not None

    def _init_layers(self):
        if self.with_du:
            self.conv_du = ConvDU(**self.du_cfg)
        if self.with_lr:
            self.conv_lr = ConvLR(**self.lr_cfg)
        if self.with_dfm:
            self.conv_dfm = DeformConvPack(**self.dfm_cfg)
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg_c = nn.Conv2d(self.feat_channels, self.num_anchors * (self.num_coords-3), 1)
        self.rpn_reg_sxy = nn.Conv2d(self.feat_channels, self.num_anchors * 3, 1)

    def init_weights(self):
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg_c, std=0.01)
        normal_init(self.rpn_reg_sxy, std=0.01)
        # todo(wangfangfang): initialize for dulr

    def forward_single(self, x):
        if self.with_du:
            x = self.conv_du(x)
        if self.with_lr:
            x = self.conv_lr(x)
        if self.with_dfm:
            x = self.conv_dfm(x)
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred_c = self.rpn_reg_c(x)
        rpn_bbox_pred_sxy = self.rpn_reg_sxy(x)
        rpn_bbox_pred = torch.cat([rpn_bbox_pred_c, rpn_bbox_pred_sxy], dim=1)
        return rpn_cls_score, rpn_bbox_pred

    def to_proposals(self, anchors, rpn_bbox_pred, scale_factor, cfg, img_shape=None):
        return NotImplementedError

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        self.print_fn('scale factor in rpn head:', scale_factor)
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            self.print_fn('rpn_head rpn_bbox_pred.shape:', rpn_bbox_pred.shape)
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, self.num_coords)
            anchors = mlvl_anchors[idx]
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
            proposals = self.to_proposals(anchors, rpn_bbox_pred, scale_factor, cfg, img_shape)

            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0] + 1
                h = proposals[:, 3] - proposals[:, 1] + 1
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) &
                                           (h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
            scores = scores.to(torch.device('cpu'))
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)
            proposals, _ = poly_soft_nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            proposals, _ = poly_soft_nms(proposals, cfg.nms_thr)  # nms_thr=0.7
            self.print_fn('proposals.shape[1]', proposals.shape[1])  # 3900+-
            proposals = proposals[:cfg.max_num, :]
            self.print_fn('before return', proposals.shape[0])  # 1000
        else:
            scores = proposals[:, -1]
            self.print_fn('proposals.shape[0]', proposals.shape[0])  # 3900+-
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]
        self.print_fn('before return', proposals.shape[0])  # 2000
        return proposals  # proposals


@HEADS.register_module
class ChebyRPNHead(CurveRPNHead, ChebyAnchorHead):
    def to_proposals(self, anchors, rpn_bbox_pred, scale_factor, cfg, img_shape=None):
        return cheby2bbox(anchors, rpn_bbox_pred, img_shape, scale_factor, self.num_coords,
                          self.target_means, self.target_stds)

@HEADS.register_module
class OffsetRPNHead(CurveRPNHead, OffsetAnchorHead):
    def to_proposals(self, anchors, rpn_bbox_pred, scale_factor, cfg, img_shape=None):
        return offset2bbox(anchors, rpn_bbox_pred, img_shape, scale_factor, 
                           self.target_means, self.target_stds)

@HEADS.register_module
class FourierRPNHead(CurveRPNHead, FourierAnchorHead):
    def to_proposals(self, anchors, rpn_bbox_pred, scale_factor, cfg, img_shape=None):
        return fori2bbox(anchors, rpn_bbox_pred, img_shape, scale_factor,
                         means=self.target_means, stds=self.target_stds,
                         sample_pts=cfg.sample_pts,
                         num_coords=self.num_coords)
    
@HEADS.register_module
class RadiusRPNHead(CurveRPNHead, RadiusAnchorHead):
    def to_proposals(self, anchors, rpn_bbox_pred, scale_factor, cfg, img_shape=None):
        return radius2bbox(anchors, rpn_bbox_pred, img_shape, scale_factor,
                           self.target_means, self.target_stds)
