from .ArT import ArTDataset
import numpy as np
import os
import mmcv
from multiprocessing import Pool
from mmdet.datasets.registry import DATASETS


@DATASETS.register_module
class ICDAR(ArTDataset):
    def load_annotations(self, ann_file):
        img_ids = mmcv.list_from_file(self.ann_file)
        self.img_ids = img_ids
        pool = Pool(12)
        img_infos = pool.map(self._load_annotations, img_ids)
        pool.close()
        pool.join()
        print("\nload success with %d samples in load_annotations" % len(img_infos))
        return img_infos

    def read_ann_info(self, img_id):
        filename = os.path.join(self.img_prefix, 'Annotations', img_id + '.gt')
        texts = open(filename).readlines()
        num_objs = len(texts)
        boxes = []
        labels = []
        centers = []
        hard_flg = np.zeros((num_objs), dtype=np.int32)

        for i in range(num_objs):
            text = texts[i].strip(' \n')
            _hard, *box = [int(i) for i in text.split(' ')]
            box = np.array(box).reshape([-1, 2])
            center = np.mean(box, axis=0)
            hard_flg[i] = _hard
            centers.append(center)
            boxes.append(box.reshape(-1))
            labels.append(1)
        return boxes, labels, centers, boxes, None, hard_flg
