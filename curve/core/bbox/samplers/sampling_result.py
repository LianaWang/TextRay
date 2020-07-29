from mmdet.utils import util_mixins


class CurveSamplingResult(util_mixins.NiceRepr):

    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, gt_cheby, gt_skeleton, assign_result,
                 gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = bboxes[pos_inds]
        self.neg_bboxes = bboxes[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds, :]
        self.pos_gt_skeleton = gt_skeleton[self.pos_assigned_gt_inds, :]
        if gt_cheby is not None:
            self.pos_gt_cheby = gt_cheby[self.pos_assigned_gt_inds, :]
        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None
