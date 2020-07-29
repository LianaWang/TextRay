from mmdet.models.detectors.rpn import RPN
from mmdet.models.registry import DETECTORS


class CurveRPN(RPN):
    def compose_rpn_loss_inputs(self,
                                gt_bboxes=None,
                                gt_coefs=None,
                                gt_skeleton=None):
        return NotImplementedError

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes=None,
                      gt_coefs=None,
                      gt_skeleton=None,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        rpn_outs = self.rpn_head(x)
        rpn_loss_inputs = rpn_outs + self.compose_rpn_loss_inputs(gt_bboxes, gt_coefs, gt_skeleton)
        losses = self.rpn_head.loss(
            *rpn_loss_inputs, gt_labels=None, img_metas=img_meta, gt_bboxes_ignore=gt_bboxes_ignore,
            cfg=self.train_cfg.rpn)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        rescale = False
        x = self.extract_feat(img)
        proposal_list = self.simple_test_rpn(x, img_meta, self.test_cfg.rpn)
        if rescale:
            for proposals, meta in zip(proposal_list, img_meta):
                proposals[:, :4] /= meta['scale_factor']
        # TODO: remove this restriction
        return proposal_list[0].cpu().numpy()


@DETECTORS.register_module
class ChebyRPN(CurveRPN):
    def compose_rpn_loss_inputs(self,
                                gt_bboxes=None,
                                gt_coefs=None,
                                gt_skeleton=None):
        return gt_bboxes, gt_coefs, gt_skeleton


@DETECTORS.register_module
class OffsetRPN(CurveRPN):
    def compose_rpn_loss_inputs(self,
                                gt_bboxes=None,
                                gt_coefs=None,
                                gt_skeleton=None):
        return gt_bboxes, gt_coefs, gt_skeleton


@DETECTORS.register_module
class PolyRPN(CurveRPN):
    def compose_rpn_loss_inputs(self,
                                gt_bboxes=None,
                                gt_coefs=None,
                                gt_skeleton=None):
        return gt_bboxes, gt_coefs, gt_skeleton


@DETECTORS.register_module
class FourierRPN(CurveRPN):
    def compose_rpn_loss_inputs(self,
                                gt_bboxes=None,
                                gt_coefs=None,
                                gt_skeleton=None):
        return gt_bboxes, gt_coefs, gt_skeleton

    
@DETECTORS.register_module
class RadiusRPN(CurveRPN):
    def compose_rpn_loss_inputs(self,
                                gt_bboxes=None,
                                gt_coefs=None,
                                gt_skeleton=None):
        return gt_bboxes, gt_coefs, gt_skeleton