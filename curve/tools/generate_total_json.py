import json
import os
from multiprocessing import Pool, cpu_count

import mmcv
import numpy as np
from shapely.geometry import Polygon

import torch
from curve.ops.polylabel import polylabel

dataset_root = '/home/chenyifeng/curve-text/data/TotalText/'
id_file = os.path.join(dataset_root, 'ImageSets/Main/test.txt')
img_ids = mmcv.list_from_file(id_file)


def _text_to_bboxes(text):
    """ 32d : [xmin, ymin, xmax, ymax, offset_x, offset_y ....]
    :param text:
    :return: polygon of 14 pts: 28d
    """
    x = text[0][5:-2].split()
    x = np.array(x, dtype=np.float32)
    y = text[1][6:-2].split()
    y = np.array(y, dtype=np.float32)
    box = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
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
        text = texts[i].split(',')
        box = _text_to_bboxes(text)
        polygon = Polygon(box.reshape((-1, 2)))
        center = np.array(polylabel(polygon)).tolist()
        bboxes.append(
            dict(
                points=box.tolist(),
                center=center,
                hard=int(text[-1][20:-3] == '#')
            )
        )
    info_dict['bboxes'] = bboxes
    return info_dict


pool = Pool(cpu_count() )
info_lists = pool.map(_convert, img_ids)
pool.close()
pool.join()
json.dump(info_lists, open(dataset_root + 'total_test_gt.json', 'w+'))
# _convert('0557')