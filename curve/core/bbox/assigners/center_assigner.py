import torch
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner


from shapely.geometry import Polygon, Point
import shapely.vectorized as sv
from curve.ops import pip_cuda # cuda version: point in polygon

class CenterAssigner(BaseAssigner):
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
                 ignore_wrt_candidates=True,
                 level_assign = False,
                 centerness_assign = False):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.level_assign = level_assign
        self.centerness_assign = centerness_assign

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
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).(anchors)
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        if bboxes.shape[0] == 0 or gt_bboxes.shape[0] == 0:
            raise ValueError('No gt or bboxes')
        inside = self.center_inside(gt_bboxes, bboxes)
       
        if self.level_assign:
            level_flags = self.level_flag(gt_bboxes, bboxes) # in: 1, out:-1
            inside = inside * level_flags
        if self.centerness_assign:
            centerness = self.compute_centerness(gt_bboxes, bboxes) #[0, 1]
            inside = (inside * centerness).clamp(max=1.0)
        if (gt_bboxes_ignore is not None) and (gt_bboxes_ignore.numel() > 0):
            ignore_insides = self.center_inside(gt_bboxes_ignore, bboxes)
            ignore_max_overlaps, _ = ignore_insides.max(dim=0)
            inside[:, ignore_max_overlaps>0] = -1
        # inside = self.center_inside_cpu(gt_bboxes, bboxes, anchor_in_gt=False, gt_in_anchor=True)
        assign_result = self.assign_wrt_inside(inside, gt_labels)

        return assign_result

    def assign_wrt_inside(self, overlaps, gt_labels=None):
        """Assign w.r.t. the overlaps of bboxes with gts.

        Args:
            overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
                shape(k, n).
            gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).
                                          None for binary classification

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

        ## for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        ## for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        # 2. assign negative: below
        neg_inds = (max_overlaps >= 0) & (max_overlaps <= self.neg_iou_thr)
        assigned_gt_inds[neg_inds] = 0

        # 3. assign positive: above positive IoU threshold
        # 4. assign ignore: an anchor contains more than one gt center
        pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_bboxes, ))
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
            ignore_inds = torch.nonzero(assigned_labels < 0).squeeze()
            assigned_gt_inds[ignore_inds] = -1
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def compute_centerness(self, gt_bboxes, anchors):
        gt_x_size = gt_bboxes[:, 0::2].max(dim=1)[0] - gt_bboxes[:, 0::2].min(dim=1)[0] + 1
        gt_y_size = gt_bboxes[:, 1::2].max(dim=1)[0] - gt_bboxes[:, 1::2].min(dim=1)[0] + 1
        img_shape = anchors[:, 2:3].max() + 1.0
        gt_bbox_size = torch.max(gt_x_size, gt_y_size)
        gt_centers = gt_bboxes[:, -2:] # last two values are center (x, y)
        # anchor centers
        anchor_size = anchors[:, 2] - anchors[:, 0] + 1
        anchor_centers = anchors.new_zeros(anchors.size(0), 2)
        anchor_centers[:, 0] = torch.mean(anchors[:, 0::2], dim=1)
        anchor_centers[:, 1] = torch.mean(anchors[:, 1::2], dim=1)
        # distance between anchor and gt
        dists = torch.norm(anchor_centers[:, None, :] - gt_centers, dim=-1) # l2 distance
        cn = torch.clamp(1.0 - 2.0 * dists / gt_bbox_size, min=0.0, max=1.0)
        return cn.t() # gt_num * anchor_num

    def level_flag(self, gt_bboxes, anchors):
        gt_x_size = gt_bboxes[:, 0::2].max(dim=1)[0] - gt_bboxes[:, 0::2].min(dim=1)[0] + 1
        gt_y_size = gt_bboxes[:, 1::2].max(dim=1)[0] - gt_bboxes[:, 1::2].min(dim=1)[0] + 1
        img_shape = anchors[:, 2:3].max() + 1.0
        gt_size = torch.max(gt_x_size, gt_y_size) / img_shape
        anchor_size = anchors[:, 2] - anchors[:, 0] + 1
        # flags
        flags = gt_bboxes.new_zeros((gt_bboxes.size(0), anchors.size(0)))
        flags[(gt_size>=0.0)&(gt_size<=0.3), :] += (anchor_size<10) * 1.0 #8
        flags[(gt_size>=0.2)&(gt_size<=0.55), :] += ((anchor_size>10)&(anchor_size<20)) * 1.0 #16
        flags[(gt_size>=0.45)&(gt_size<=0.8), :] += ((anchor_size>25)&(anchor_size<36)) * 1.0 #32
        flags[(gt_size>=0.7), :] += ((anchor_size>50)&(anchor_size<70)) * 1.0 #64
#       flags[gt_size<=0.25, :] += (anchor_size<10) * 1.0 #8
#       flags[(gt_size>0.15)&(gt_size<=0.45), :] += ((anchor_size>10)&(anchor_size<20)) * 1.0 #16
#       flags[(gt_size>0.35)&(gt_size<=0.65), :] += ((anchor_size>25)&(anchor_size<36)) * 1.0 #32
#       flags[(gt_size>0.55)&(gt_size<=0.85), :] += ((anchor_size>50)&(anchor_size<70)) * 1.0 #64
#       flags[gt_size>0.75, :] += (anchor_size>100) * 1.0 #128
        # -1 or +1
        flags[flags<1] = -1
        flags[flags>=1] = 1
        return flags
    
    def center_inside_cpu(self, gt_bboxes, anchors, anchor_in_gt=True, gt_in_anchor=True):
        gt_centers = torch.round(gt_bboxes[:, -2:])
        gt_bboxes = gt_bboxes[:, :-2] # remove gt_centers
        intersects = gt_bboxes.new_zeros((gt_bboxes.size(0), anchors.size(0)))
        if gt_in_anchor: # compute gt center in anchor
            left_top = torch.sum(gt_centers[:, None] >= anchors[:, :2], dim=-1)
            right_bottom = torch.sum(gt_centers[:, None] <= anchors[:, -2:], dim=-1)
            intersects += ((left_top + right_bottom) == 4).float()
        # anchor center in gt
        if anchor_in_gt:
            intersects = intersects.cpu()
            gt_bboxes = gt_bboxes.cpu()
            anchor_center_x = torch.mean(anchors[:, 0::2], dim=1).cpu()
            anchor_center_y = torch.mean(anchors[:, 1::2], dim=1).cpu()
            for k in range(len(gt_bboxes)):
                gt_bbox = Polygon(gt_bboxes[k, :].reshape((-1, 2)))
                inside_gt = sv.contains(gt_bbox, x=anchor_center_x, y=anchor_center_y)
                intersects[k, :] += torch.FloatTensor(inside_gt) # 0 or 1
        return intersects.cuda()
 
    def center_inside(self, gt_bboxes, anchors, gt_in_anchor=False):
        gt_centers = torch.round(gt_bboxes[:, -2:])
        gt_bboxes = gt_bboxes[:, :-2] # remove gt_centers
        anchor_centers = anchors.new_zeros(anchors.size(0), 2)
        anchor_centers[:, 0] = torch.mean(anchors[:, 0::2], dim=1)
        anchor_centers[:, 1] = torch.mean(anchors[:, 1::2], dim=1)
        insides = pip_cuda(gt_bboxes, anchor_centers)
        if gt_in_anchor: # compute gt center in anchor
            left_top = torch.sum(gt_centers[:, None] >= anchors[:, :2], dim=-1)
            right_bottom = torch.sum(gt_centers[:, None] <= anchors[:, -2:], dim=-1)
            insides += ((left_top + right_bottom) == 4).float() * 10.0
        return insides
