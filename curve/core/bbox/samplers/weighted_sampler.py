import numpy as np
import torch

from .base_sampler import CurveBaseSampler


class CurveWeightedSampler(CurveBaseSampler):
    def __init__(self,
                 num,
                 pos_fraction,
                 floor_thr=-1,
                 floor_fraction=0,
                 num_bins=3,
                 **kwargs):
        super(CurveWeightedSampler, self).__init__(num, pos_fraction,
                                                    **kwargs)

    def random_choice(self, gallery, num, weights):
        assert len(gallery) >= num
        # cuda random sampling is too slow
        rand_inds = torch.multinomial(weights.cpu(), num).unique()
        assert len(rand_inds) == num, "centerness sampler rand inds error"
        return gallery[rand_inds.to(gallery.device)]

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Randomly sample some positive samples."""
        pos_inds = torch.nonzero(assign_result.gt_inds > 0)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            overlaps = assign_result.max_overlaps[pos_inds]
            sampled_inds = self.random_choice(pos_inds, num_expected, overlaps)
            return sampled_inds


    def _sample_neg(self, assign_result, num_expected, **kwargs):
        neg_inds = torch.nonzero(assign_result.gt_inds == 0)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            overlaps = assign_result.max_overlaps[neg_inds]
            weights = overlaps + 1e-3 # avoid 0 or too small prob, not need to sum to 1
            sampled_inds = self.random_choice(neg_inds, num_expected, weights)
            return sampled_inds
