from .. import pip_cuda as pip
from .. import iou_cuda as iou

def iou_cuda(boxes):
    return iou.iou(boxes)

def pip_cuda(boxes, points):
    return pip.pip(boxes, points)