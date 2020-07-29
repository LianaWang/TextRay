import json
import os
from multiprocessing import Pool

import mmcv
import numpy as np
from shapely.geometry import Polygon

from curve.ops.polylabel import polylabel

dataset_root = '/home/chenyifeng/curve-text/data/CTW1500/'
id_file = os.path.join(dataset_root, 'ImageSets/Main/test.txt')
img_ids = mmcv.list_from_file(id_file)


def _text_to_bboxes(text):
    """ 32d : [xmin, ymin, xmax, ymax, offset_x, offset_y ....]
    :param text:
    :return: polygon of 14 pts: 28d
    """
    box = text.split(',')  #
    box = np.array(box, dtype=np.int32)
    box[4::2] += box[0]
    box[5::2] += box[1]
    box = box[4:]
    return box


def _convert(img_id):
    im_path = os.path.join(dataset_root, 'JPGImages/{}.jpg'.format(img_id))
    height, width, _ = mmcv.imread(im_path).shape
    info_dict = dict(image_id=img_id, width=width, height=height)
    filename = os.path.join(dataset_root, 'Annotations', img_id + '.txt')
    texts = open(filename).readlines()

    num_objs = len(texts)
    bboxes = []

    for i in range(num_objs):
        box = _text_to_bboxes(texts[i])
        polygon = Polygon(box.reshape((-1, 2)))
        center = np.array(polylabel(polygon))
        bboxes.append(
            dict(
                points=box.tolist(),
                center=center.tolist(),
                hard=0
            )
        )

    info_dict["bboxes"] = bboxes
    return info_dict


pool = Pool(8)
info_lists = pool.map(_convert, img_ids)
pool.close()
pool.join()
json.dump(info_lists, open(dataset_root + 'ctw_test_gt.json', 'w+'))
