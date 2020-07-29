from .ArT import ArTDataset
from .CTW1500 import CTW1500
from .TotalText import TotalText
from .ICDAR import ICDAR
from .MSRA import MSRA
from .pipelines import *
from mmdet.datasets.cityscapes import CityscapesDataset
from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.dataset_wrappers import ConcatDataset, RepeatDataset
from mmdet.datasets.loader import DistributedGroupSampler, GroupSampler, build_dataloader
from mmdet.datasets.registry import DATASETS
from mmdet.datasets.voc import VOCDataset
from mmdet.datasets.wider_face import WIDERFaceDataset
from mmdet.datasets.xml_style import XMLDataset
from mmdet.datasets.builder import build_dataset

__all__ = [
    'CustomDataset', 'XMLDataset', 'CocoDataset', 'VOCDataset',
    'CityscapesDataset', 'GroupSampler', 'DistributedGroupSampler',
    'build_dataloader', 'ConcatDataset', 'RepeatDataset', 'WIDERFaceDataset',
    'DATASETS', 'build_dataset'
]