import os
from multiprocessing import Pool

import mmcv
import numpy as np
from shapely.geometry import Polygon

from curve.ops.polylabel import polylabel
from mmdet.datasets.registry import DATASETS
from .ArT import ArTDataset
from .curve_utils import expand_twelve


@DATASETS.register_module
class TotalText(ArTDataset):
    def load_annotations(self, ann_file):
        img_ids = mmcv.list_from_file(self.ann_file)
        self.img_ids = img_ids
        pool = Pool(12)
        img_infos = pool.map(self._load_annotations, img_ids)
        pool.close()
        pool.join()
        print("\nload success with %d samples in load_annotations" % len(img_infos))
        return img_infos

    def _text_to_bboxes(self, text):
        x = text[0][5:-2].split()
        x = np.array(x, dtype=np.float32)
        y = text[1][6:-2].split()
        y = np.array(y, dtype=np.float32)
        box = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
        return box

    def read_ann_info(self, img_id):
        filename = os.path.join(self.img_prefix, 'Annotations', img_id + '.txt')
        texts = open(filename).readlines()
        num_objs = len(texts)
        points = []
        boxes = []
        labels = []
        centers = []
        hard_flag = np.zeros((num_objs), dtype=np.int32)

        for i in range(num_objs):
            text = texts[i].split(',')
            pts = self._text_to_bboxes(text)
            hard = int(text[-1][20:-3] == '#')
            twelve_pts = expand_twelve(pts)
            center = np.array(polylabel(Polygon(pts)))

            points.append(pts)
            boxes.append(twelve_pts.reshape(-1))
            centers.append(center)
            labels.append(1)
            hard_flag[i] = hard

        return boxes, labels, centers, points, None, hard_flag
