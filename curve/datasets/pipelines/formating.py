import numpy as np
from mmcv.parallel import DataContainer as DC

from mmdet.datasets.pipelines import to_tensor
from mmdet.datasets.registry import PIPELINES

__all__ = ['ToCurveDataContainer', 'DefaultCurveFormatBundle']

@PIPELINES.register_module
class ToCurveDataContainer(object):

    def __init__(self,
                 fields=(dict(key='img', stack=True), dict(key='gt_bboxes'), dict(key='gt_bboxes_ignore'),
                         dict(key='gt_coefs'), dict(key='gt_skeleton'), dict(key='gt_labels'))):
        self.fields = fields

    def __call__(self, results):
        for field in self.fields:
            field = field.copy()
            key = field.pop('key')
            results[key] = DC(results[key], **field)
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(fields={})'.format(self.fields)


@PIPELINES.register_module
class DefaultCurveFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        if 'img' in results:
            img = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        for key in ['proposals', 'gt_bboxes', 'gt_coefs', 'gt_skeleton', 'gt_bboxes_ignore', 'gt_labels']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)
        return results

    def __repr__(self):
        return self.__class__.__name__
