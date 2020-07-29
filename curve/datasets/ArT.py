import json
import os.path as osp
from multiprocessing import Pool
import os
import mmcv
import numpy as np
from shapely.geometry import Polygon

from curve.ops.polylabel import polylabel
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.registry import DATASETS
from .curve_utils import expand_twelve, sample_contour, cheby_fit, poly_fit, fourier_fit, rotate_cheby_fit


@DATASETS.register_module
class ArTDataset(CustomDataset):
    CLASSES = ('text')

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 cache_root=None,
                 img_prefix=None,
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 debug_mode=False,
                 encoding='cheby',
                 degree=22,
                 sample_pts=360):
        self.data_root = data_root
        self.cache_root = cache_root
        self.img_prefix = img_prefix
        self.degree = degree
        self.sample_pts = sample_pts
        # join paths if data_root is specified
        if self.data_root is not None:
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
        assert encoding in ['cheby', 'poly', 'fourier', 'none']
        self.encoding = encoding
        self.debug = debug_mode
        # super init
        super(ArTDataset, self).__init__(ann_file, pipeline, data_root,
                                         img_prefix, seg_prefix,
                                         proposal_file, test_mode)

    def load_annotations(self, ann_file):
        img_ids = mmcv.list_from_file(self.ann_file)
        self.img_ids = img_ids
        anno_path = osp.join(self.img_prefix, 'train_labels.json')
        with open(anno_path, 'r') as f:
            self.load_dict = json.load(f)
        if self.debug:
            print('Skipping Loading Anoonations')
            return [] * len(img_ids)
        dir_name = osp.join(self.img_prefix, self.cache_root) #'Cache_%dx%d'%(scale[0], scale[1])
        if not osp.exists(dir_name):
            os.makedirs(dir_name)
        pool = Pool(8)
        img_infos = pool.map(self._load_annotations, img_ids)
        pool.close()
        pool.join()
        print("\nload success with %d samples in load_annotations" % len(img_infos))
        return img_infos

    def _load_annotations(self, img_id):
        dir_name = osp.join(self.img_prefix, self.cache_root)
        ann_path = '{}/{}_{}_{}.npy'.format(dir_name, img_id, self.encoding, self.degree)
        try:
            info_dict = np.load(ann_path, allow_pickle=True).item()
        except Exception as err:
            if osp.exists(ann_path):
                print(err)
            filename = 'JPGImages/{}.jpg'.format(img_id)
            im_path = osp.join(self.img_prefix, filename)
            height, width, _ = mmcv.imread(im_path).shape
            info_dict = dict(id=img_id, filename=filename, width=width, height=height)
            if not self.test_mode:
                ann = self.compute_ann_info(img_id)
                info_dict.update({"ann": ann})
            np.save(ann_path, info_dict)
            print('.', end='', flush=True)
        return info_dict

    def read_ann_info(self, img_id):
        """
        boxes are list of 1d coordinates, labels are all ones, centers are center points of polygons
        points are original pts of [k, 2]
        trans, hard_flag ...
        :param img_id:
        :return: boxes, labels, centers, points, trans, hard_flg
        """
        ano = self.load_dict['gt_%d' % (int(img_id))]
        boxes = []
        labels = []
        centers = []
        points = []
        trans = []
        num_objs = len(ano)
        hard_flg = np.zeros((num_objs), dtype=np.int32)

        for ix in range(num_objs):
            pts = np.array(ano[ix]['points'])
            twelve_pts = expand_twelve(pts)
            center = polylabel(Polygon(pts))
            points.append(pts)
            centers.append(np.array(center))
            boxes.append(twelve_pts.reshape(-1))
            labels.append(1)
            trans.append(len(ano[ix]['transcription']))
            hard_flg[ix] = 1 if ano[ix]['illegibility'] else 0
        return boxes, labels, centers, points, trans, hard_flg

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        if ann_info is None: # remove shits
            return None
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def compute_ann_info(self, img_id):
        boxes, labels, centers, points, _, hard_flg = self.read_ann_info(img_id)
        # from points, centers to extract 360 contour points
        contours, skeleton_ori, hard_flg = sample_contour(points, centers, hard_flg, self.sample_pts)
        if len(contours) == 0:
            return None

        skeleton = np.array(contours)[:, :, 1].reshape((-1, self.sample_pts))
        idx_ignore = np.where(hard_flg == 1)[0]
        idx_easy = np.where(hard_flg == 0)[0]
        boxes_easy = np.array(boxes)[idx_easy, :]
        boxes_ignore = np.array(boxes)[idx_ignore, :]
        centers_easy = np.array(centers)[idx_easy, :]
        centers_ignore = np.array(centers)[idx_ignore, :]
        labels = np.array(labels)[idx_easy]
        try:
            bboxes = np.hstack((boxes_easy, centers_easy))
        except:
            print(boxes_easy.shape, centers_easy.shape)
        bboxes_ignore = np.hstack((boxes_ignore, centers_ignore))

        ann = dict(
            bboxes=bboxes.astype(np.float32),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels=labels.astype(np.int64),
            skeleton=skeleton.astype(np.float32),
            skeleton_ori=skeleton_ori
        )
        
        if self.encoding == 'cheby':
            cheby_coef = cheby_fit(contours, self.degree)
            assert idx_easy.shape[0] == cheby_coef.shape[0]  
            cheby = np.hstack((cheby_coef, centers_easy))
            ann.update({"cheby": cheby.astype(np.float32)})
        elif self.encoding == 'fourier':
            fori_coef = fourier_fit(contours, self.degree)
            assert idx_easy.shape[0] == fori_coef.shape[0]
            fori = np.hstack((fori_coef, centers_easy))
            ann.update({"fourier": fori.astype(np.float32)})
        elif self.encoding == 'poly':
            poly_coef = poly_fit(contours, self.degree)
            assert idx_easy.shape[0] == poly_coef.shape[0]  
            poly = np.hstack((poly_coef, centers_easy))
            ann.update({"poly": poly.astype(np.float32)})
#         elif self.encoding == 'rotate_cheby':
#             rotate_cheby_coef = rotate_cheby_fit(contours, skeleton_ori, self.degree)
#             assert idx_easy.shape[0] == rotate_cheby_coef.shape[0]  
#             rotate_cheby = np.hstack((rotate_cheby_coef, centers_easy))
#             ann.update({"rotate_cheby": rotate_cheby.astype(np.float32)})
#         elif self.encoding == 'rotate_fourier':
#             rotate_fori_coef = rotate_fourier_fit(contours, skeleton_ori, self.degree/2)
#             assert idx_easy.shape[0] == rotate_fori_coef.shape[0]
#             rotate_fori = np.hstack((rotate_fori_coef, centers_easy))
#             ann.update({"rotate_fourier": fori.astype(np.float32)})

        return ann
