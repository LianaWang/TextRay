from mmdet.core.bbox.samplers import RandomSampler
from .base_sampler import CurveBaseSampler


class CurveRandomSampler(RandomSampler, CurveBaseSampler):

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        super(CurveRandomSampler, self).__init__(num, pos_fraction, neg_pos_ub,
                                                 add_gt_as_proposals)
