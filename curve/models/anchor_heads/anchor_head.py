from __future__ import division

import torch
import torch.nn as nn

from torch import flatten
from curve.core.anchor import cheby_anchor_target, offset_anchor_target, fourier_anchor_target, radius_anchor_target
from mmdet.core import (force_fp32, multi_apply)
from mmdet.models.anchor_heads.anchor_head import AnchorHead
from mmdet.models.builder import build_loss
from mmdet.models.registry import HEADS


class CurveAnchorHead(AnchorHead):
    def __init__(self,
                 num_coords=26,
                 loss_ctr=None,
                 verbose=False,
                 **kwargs):
        self.num_coords = num_coords
        self.print_fn = print if verbose else lambda *x: 0
        super(CurveAnchorHead, self).__init__(**kwargs)
        if loss_ctr is not None:
            self.loss_ctr = build_loss(loss_ctr)
            

    @property
    def with_ctr_loss(self):
        return hasattr(self, 'loss_ctr') and self.loss_ctr is not None


    def _init_layers(self):
        self.conv_cls = nn.Conv2d(self.feat_channels,
                                  self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels, self.num_anchors * self.num_coords, 1)


@HEADS.register_module
class ChebyAnchorHead(CurveAnchorHead):
    def loss_single(self, cls_score, bbox_pred,
                    labels, label_weights,
                    bbox_targets, bbox_weights,
                    ctr_targets, ctr_weights,
                    num_total_samples, num_total_pos, cfg):
        """ all gts are in [bs, h*w, d]
        :param cls_score:
        :param bbox_pred:
        :param labels:
        :param label_weights:
        :param bbox_targets:
        :param bbox_weights:
        :param ctr_targets:
        :param ctr_weights:
        :param num_total_samples:
        :param cfg:
        :return:
        """
        # classification loss
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, flatten(labels), flatten(label_weights), avg_factor=num_total_samples)
        # regression loss
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, self.num_coords)
        loss_bbox = self.loss_bbox(
            bbox_pred[:, :self.num_coords-3],
            flatten(bbox_targets, 0, -2),  # hide batch
            flatten(bbox_weights, 0, -2),  # hide batch
            avg_factor=num_total_pos * bbox_weights.size(-1))
        losses = (loss_cls, loss_bbox)
        if self.with_ctr_loss:
            loss_ctr = self.loss_ctr(
                bbox_pred[:, -3:],
                flatten(ctr_targets, 0, -2),
                flatten(ctr_weights, 0, -2),
                avg_factor=num_total_pos * ctr_weights.size(-1))
            losses += (loss_ctr, )
        return losses


    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_cheby,
             gt_skeleton,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = cheby_anchor_target(
            anchor_list,
            valid_flag_list,
            bbox_preds,
            gt_bboxes,
            gt_cheby,
            gt_skeleton,
            img_metas,
            self.target_means,
            self.target_stds,
            self.num_coords,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        if cls_reg_targets is None:
            return None

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         ctr_targets_list, ctr_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        losses = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            ctr_targets_list,
            ctr_weights_list,
            num_total_samples=num_total_samples,
            num_total_pos=num_total_pos,
            cfg=cfg)
        loss_dict = dict(loss_cls=losses[0], loss_bbox=losses[1])
        if self.with_ctr_loss:
            loss_dict.update({'loss_ctr': losses[2]})
        return loss_dict


@HEADS.register_module
class OffsetAnchorHead(CurveAnchorHead):
    @property
    def with_ctr_loss(self):
        return False

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, num_total_pos, cfg):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, self.num_coords)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            flatten(bbox_targets, 0, -2),
            flatten(bbox_weights, 0, -2),
            avg_factor=num_total_pos*bbox_weights.size(-1))
        # end debug
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_cheby, # not used
             gt_skeleton,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = offset_anchor_target(
            anchor_list,
            valid_flag_list,
            bbox_preds,
            gt_bboxes,
            None,
            gt_skeleton,
            img_metas,
            self.target_means,
            self.target_stds,
            self.num_coords,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        if cls_reg_targets is None:
            return None

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            num_total_pos=num_total_pos,
            cfg=cfg)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)


@HEADS.register_module
class FourierAnchorHead(ChebyAnchorHead):
    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_cheby,
             gt_skeleton,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = fourier_anchor_target(
            anchor_list,
            valid_flag_list,
            bbox_preds,
            gt_bboxes,
            gt_cheby,
            gt_skeleton,
            img_metas,
            self.target_means,
            self.target_stds,
            self.num_coords,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        if cls_reg_targets is None:
            return None

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         ctr_targets_list, ctr_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        losses = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            ctr_targets_list,
            ctr_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg)
        loss_dict = dict(loss_cls=losses[0], loss_bbox=losses[1])
        if self.with_ctr_loss:
            loss_dict.update({'loss_ctr': losses[2]})
        return loss_dict

    def loss_single(self, cls_score, bbox_pred,
                    labels, label_weights,
                    bbox_targets, bbox_weights,
                    ctr_targets, ctr_weights,
                    num_total_samples, cfg):
        """ all gts are in [bs, h*w, d]
        :param cls_score:
        :param bbox_pred:
        :param labels:
        :param label_weights:
        :param bbox_targets:
        :param bbox_weights:
        :param ctr_targets:
        :param ctr_weights:
        :param num_total_samples:
        :param cfg:
        :return:
        """
        # classification loss
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, flatten(labels), flatten(label_weights), avg_factor=num_total_samples)
        # regression loss
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, self.num_coords)
        loss_bbox = self.loss_bbox(
            bbox_pred[:, :self.num_coords - 3],
            flatten(bbox_targets, 0, -2),  # hide batch
            flatten(bbox_weights, 0, -2),  # hide batch
            avg_factor=num_total_samples)
        losses = (loss_cls, loss_bbox)
        if self.with_ctr_loss:
            loss_ctr = self.loss_ctr(
                bbox_pred[:, - 3:],
                flatten(ctr_targets, 0, -2),
                flatten(ctr_weights, 0, -2),
                avg_factor=num_total_samples)
            losses += (loss_ctr, )
        return losses
    
    
@HEADS.register_module
class RadiusAnchorHead(CurveAnchorHead):
    def loss_single(self, cls_score, bbox_pred,
                    labels, label_weights,
                    bbox_targets, bbox_weights,
                    ctr_targets, ctr_weights,
                    num_total_samples, num_total_pos, cfg):
        """ all gts are in [bs, h*w, d]
        :param cls_score:
        :param bbox_pred:
        :param labels:
        :param label_weights:
        :param bbox_targets:
        :param bbox_weights:
        :param ctr_targets:
        :param ctr_weights:
        :param num_total_samples:
        :param cfg:
        :return:
        """
        # classification loss
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, flatten(labels), flatten(label_weights), avg_factor=num_total_samples)
        # regression loss
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, self.num_coords)
        loss_bbox = self.loss_bbox(
            bbox_pred[:, :self.num_coords-3],
            flatten(bbox_targets, 0, -2),  # hide batch
            flatten(bbox_weights, 0, -2),  # hide batch
            avg_factor=num_total_pos*bbox_weights.size(-1))
        losses = (loss_cls, loss_bbox)
        if self.with_ctr_loss:
            loss_ctr = self.loss_ctr(
                bbox_pred[:, -3:],
                flatten(ctr_targets, 0, -2),
                flatten(ctr_weights, 0, -2),
                avg_factor=num_total_pos*ctr_weights.size(-1))
            losses += (loss_ctr, )
        return losses


    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_cheby,
             gt_skeleton,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = radius_anchor_target(
            anchor_list,
            valid_flag_list,
            bbox_preds,
            gt_bboxes,
            gt_cheby,
            gt_skeleton,
            img_metas,
            self.target_means,
            self.target_stds,
            self.num_coords,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        if cls_reg_targets is None:
            return None

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         ctr_targets_list, ctr_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        losses = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            ctr_targets_list,
            ctr_weights_list,
            num_total_samples=num_total_samples,
            num_total_pos=num_total_pos,
            cfg=cfg)
        loss_dict = dict(loss_cls=losses[0], loss_bbox=losses[1])
        if self.with_ctr_loss:
            loss_dict.update({'loss_ctr': losses[2]})
        return loss_dict
