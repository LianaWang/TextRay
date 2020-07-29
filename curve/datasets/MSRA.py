import os
from multiprocessing import Pool

import mmcv
import numpy as np

from mmdet.datasets.registry import DATASETS
from .ArT import ArTDataset
from .curve_utils import expand_twelve
import os.path as osp


@DATASETS.register_module
class MSRA(ArTDataset):
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
        filename = os.path.join(self.img_prefix, 'Annotations', 'IMG_{:04d}.gt'.format(int(img_id)))
        texts = open(filename).readlines()
        num_objs = len(texts)
        points = []
        boxes = []
        labels = []
        centers = []
        hard_flag = np.zeros((num_objs), dtype=np.int32)

        for i in range(num_objs):
            text = texts[i].rstrip(' \n')
            *raw, theta = text.split()
            _, hard, x, y, w, h = [int(i) for i in raw]  # first element is raw object index
            theta = float(theta)

            offsets = np.array([[-w / 2.0, -h / 2.0], [w / 2.0, -h / 2.0], [w / 2.0, h / 2.0], [-w / 2.0, h / 2.0]])
            rotate = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
            offsets = offsets.dot(rotate)
            center = np.array([x + w / 2.0, y + h / 2.0])
            pts = center[np.newaxis, :] + offsets
            twelve_pts = expand_twelve(pts)
            points.append(pts)
            boxes.append(twelve_pts.reshape(-1))
            centers.append(center)
            labels.append(1)
            hard_flag[i] = int(hard)

        return boxes, labels, centers, points, None, hard_flag

    def _load_annotations(self, img_id):
        dir_name = osp.join(self.img_prefix, self.cache_root)
        ann_path = '{}/{}_{}.npy'.format(dir_name, img_id, self.encoding)
        try:
            info_dict = np.load(ann_path, allow_pickle=True).item()
        except Exception as err:
            if osp.exists(ann_path):
                print(err)
            filename = 'JPGImages/IMG_{:04d}.JPG'.format(int(img_id))
            im_path = osp.join(self.img_prefix, filename)
            height, width, _ = mmcv.imread(im_path).shape
            info_dict = dict(id=img_id, filename=filename, width=width, height=height)
            if not self.test_mode:
                ann = self.compute_ann_info(img_id)
                info_dict.update({"ann": ann})
            np.save(ann_path, info_dict)
            print('.', end='', flush=True)
        return info_dict
