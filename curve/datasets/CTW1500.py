import os
from multiprocessing import Pool

import mmcv
import numpy as np
from shapely.geometry import Polygon

from curve.ops.polylabel import polylabel
from mmdet.datasets.registry import DATASETS
from .ArT import ArTDataset


@DATASETS.register_module
class CTW1500(ArTDataset):
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
        """ 32d : [xmin, ymin, xmax, ymax, offset_x, offset_y ....]
        :param text:
        :return: polygon of 14 pts: 28d
        """
        box = text.split(',')  #
        box = np.array(box, dtype=np.float32)
        box[4::2] += box[0]
        box[5::2] += box[1]
        box = box[4:]
        return box

    def read_ann_info(self, img_id):
        filename = os.path.join(self.img_prefix, 'Annotations', img_id + '.txt')
        texts = open(filename).readlines()
        num_objs = len(texts)
        boxes = []
        labels = []
        centers = []
        hard_flg = np.zeros((num_objs), dtype=np.int32)

        for i in range(num_objs):
            box = self._text_to_bboxes(texts[i])
            polygon = Polygon(box.reshape((-1, 2)))
            centers.append(np.array(polylabel(polygon)))
            boxes.append(box)
            labels.append(1)
        return boxes, labels, centers, boxes, None, hard_flg
