import torch
import numpy as np
from mmdet.core.bbox.geometry import bbox_overlaps
from curve.ops import bbox_overlap_cpu
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner
from shapely.geometry import Polygon


class MaxIoUAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index.

    - -1: don't care
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
    """

    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, 0, or a positive number. -1 means don't care,
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to -1
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        if bboxes.shape[0] == 0 or gt_bboxes.shape[0] == 0:
            raise ValueError('No gt or bboxes')
        # minx, _ = torch.min(gt_bboxes[:, 0::2], dim=1)
        # miny, _ = torch.min(gt_bboxes[:, 1::2], dim=1)
        # maxx, _ = torch.max(gt_bboxes[:, 0::2], dim=1)
        # maxy, _ = torch.max(gt_bboxes[:, 1::2], dim=1)
        # gt_bboxes = torch.stack([minx, miny, maxx, maxy], dim = 1)
        
        # overlaps = self.bbox_inter(gt_bboxes, bboxes)
        gt_bboxes = gt_bboxes[:, :24].to('cpu')
        bboxes = bboxes.to('cpu')
        overlaps = bbox_overlap_cpu.inter(gt_bboxes, bboxes)    #self.bbox_inter
        # print(overlaps.max())
        # if (self.ignore_iof_thr > 0) and (gt_bboxes_ignore is not None) and (
        #         gt_bboxes_ignore.numel() > 0):
        #     if self.ignore_wrt_candidates:
        #         ignore_overlaps = bbox_overlaps(
        #             bboxes, gt_bboxes_ignore, mode='iof')
        #         ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
        #     else:
        #         ignore_overlaps = bbox_overlaps(
        #             gt_bboxes_ignore, bboxes, mode='iof')
        #         ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
        #     overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
        return assign_result

    def assign_wrt_overlaps(self, overlaps, gt_labels=None):
        """Assign w.r.t. the overlaps of bboxes with gts.

        Args:
            overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
                shape(k, n).
            gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        if overlaps.numel() == 0:
            raise ValueError('No gt or proposals')

        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        # 2. assign negative: below
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                             & (max_overlaps < self.neg_iou_thr[1])] = 0

        # 3. assign positive: above positive IoU threshold
        pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        # 4. assign fg: for each gt, proposals with highest IoU
        for i in range(num_gts):
            if gt_max_overlaps[i] > self.min_pos_iou:
                if self.gt_max_assign_all:
                    max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                    assigned_gt_inds[max_iou_inds] = i + 1
                else:
                    assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_bboxes, ))
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        # print('assigned indexes', overlaps.size(), torch.sum(assigned_gt_inds>0))
        # print('labels', assigned_labels)

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def bbox_inter(self, gt_bboxes, anchors):
        gt_bboxes = gt_bboxes[:, :24]
        intersects = torch.zeros((gt_bboxes.size(0), anchors.size(0)))
        for k in range(len(gt_bboxes)):
            x_min = torch.min(gt_bboxes[k, 0::2])
            y_min = torch.min(gt_bboxes[k, 1::2])
            x_max = torch.max(gt_bboxes[k, 0::2])
            y_max = torch.max(gt_bboxes[k, 1::2])
            # print(x_min, y_min, x_max, y_max)
            inter_flag = ((x_min < anchors[:, 2]) & (y_min < anchors[:, 3]) &
                          (x_max > anchors[:, 0]) & (y_max > anchors[:, 1]))
            # print(inter_flag, inter_flag.size())
            # print(torch.where(inter_flag>0)[0])
            gt_bbox = Polygon(gt_bboxes[k, :].reshape((-1, 2))).buffer(0)
            for n in map(int, torch.where(inter_flag>0)[0]):
                anchor = Polygon([[anchors[n,0], anchors[n,1]], [anchors[n,2], anchors[n,1]], [anchors[n,2], anchors[n,3]], [anchors[n,0], anchors[n,3]]])
                try:
                    inter = gt_bbox.intersection(anchor)
                    intersects[k,n] = inter.area / anchor.area if anchor.area>0 else 1
                except:
                    intersects[k,n] = 0
        return torch.Tensor(intersects)
    

