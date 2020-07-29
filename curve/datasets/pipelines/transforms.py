import mmcv
import numpy as np
from numpy import random
import cv2
import math
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.datasets.registry import PIPELINES

__all__ = ['CurveResize', 'CurveRandomCrop', 'CurveRandomFlip', 'CurveSegResizeFlipPadRescale',
           'CurveExpand', 'CurvePad', 'CurveMinIoURandomCrop']


@PIPELINES.register_module
class CurveResize(object):
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:
    - `ratio_range` is not None: randomly sample a ratio from the ratio range
        and multiply it with the image scale.
    - `ratio_range` is None and `multiscale_mode` == "range": randomly sample a
        scale from the a range.
    - `ratio_range` is None and `multiscale_mode` == "value": randomly sample a
        scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        # print('resizing img, scale:', results['scale'])
        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(
                results['img'], results['scale'], return_scale=True)
        else:
            img, w_scale, h_scale = mmcv.imresize(
                results['img'], results['scale'], return_scale=True)
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        img_shape = results['img_shape']
        # print('img_shape:', img_shape)
        # print('resizing box, scale_factor in transforms:', results['scale_factor'])
        # print('results',results)
        for key in ['gt_bboxes', 'gt_bboxes_ignore', 'gt_skeleton']:
            if key in results.get('bbox_fields', []):
                bboxes = results[key] * results['scale_factor']
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
                results[key] = bboxes
        for key in ['gt_coefs']:
            if key in results.get('bbox_fields', []):
                bboxes = results[key]
                bboxes[:, -3:] = bboxes[:, -3:] * results['scale_factor']
                results[key] = bboxes

    def _resize_masks(self, results):
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            if self.keep_ratio:
                masks = [
                    mmcv.imrescale(
                        mask, results['scale_factor'], interpolation='nearest')
                    for mask in results[key]
                ]
            else:
                mask_size = (results['img_shape'][1], results['img_shape'][0])
                masks = [
                    mmcv.imresize(mask, mask_size, interpolation='nearest')
                    for mask in results[key]
                ]
            results[key] = masks

    def __call__(self, results):
        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(img_scale={}, multiscale_mode={}, ratio_range={}, '
                     'keep_ratio={})').format(self.img_scale,
                                              self.multiscale_mode,
                                              self.ratio_range,
                                              self.keep_ratio)
        return repr_str


@PIPELINES.register_module
class CurveRandomFlip(object):
    """Flip the image & bbox & mask.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio (float, optional): The flipping probability.
    """

    def __init__(self, flip_ratio=None):
        self.flip_ratio = flip_ratio
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1

    def bbox_flip(self, bboxes, img_shape):
        """Flip bboxes horizontally.

        Args:
            bboxes(ndarray): shape (..., 4*k)
            img_shape(tuple): (height, width)
        """
        # print(bboxes.shape) 6*26
        assert bboxes.shape[-1] % 4 == 0

        w = img_shape[1]
        flipped = bboxes.copy()
        flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
        flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
        return flipped

    def __call__(self, results):
        if 'flip' not in results:
            flip = True if np.random.rand() < self.flip_ratio else False
            results['flip'] = flip
        if results['flip']:
            # flip image
            results['img'] = mmcv.imflip(results['img'])
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'])
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = [mask[:, ::-1] for mask in results[key]]
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(flip_ratio={})'.format(
            self.flip_ratio)


@PIPELINES.register_module
class CurvePad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        if self.size is not None:
            padded_img = mmcv.impad(results['img'], self.size)
        elif self.size_divisor is not None:
            padded_img = mmcv.impad_to_multiple(
                results['img'], self.size_divisor, pad_val=self.pad_val)
        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        pad_shape = results['pad_shape'][:2]
        for key in results.get('mask_fields', []):
            padded_masks = [
                mmcv.impad(mask, pad_shape, pad_val=self.pad_val)
                for mask in results[key]
            ]
            results[key] = np.stack(padded_masks, axis=0)

    def __call__(self, results):
        self._pad_img(results)
        self._pad_masks(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(size={}, size_divisor={}, pad_val={})'.format(
            self.size, self.size_divisor, self.pad_val)
        return repr_str

@PIPELINES.register_module
class CurveRandomCrop(object):
    """Random crop the image & bboxes & masks.

    Args:
    """

    def __init__(self, final_size=800, scale_range=(0.5,2.0)):
        self.final_size = final_size * 1.0
        self.down_scale, self.up_scale = scale_range

    def __call__(self, results):
        img = results['img']
        origin_shape = img.shape
        ignore_size = 8.0
        
        bboxes = results['gt_bboxes']
        for try_bbox_idx in range(20):
            cur_box_idx = np.random.randint(0, bboxes.shape[0])
            min_x = np.int32(np.min(bboxes[cur_box_idx, 0::2]))
            min_y = np.int32(np.min(bboxes[cur_box_idx, 1::2]))
            max_x = np.int32(np.max(bboxes[cur_box_idx, 0::2]))
            max_y = np.int32(np.max(bboxes[cur_box_idx, 1::2]))
            bbox_shape = [max_x - min_x + 1.0, max_y - min_y + 1.0]
            # margin
            upscale = self.up_scale
            downscale = max(self.down_scale, (2.0 * ignore_size)/min(bbox_shape)) # do not downsample bbox too small
            small_m = max(32, int(self.final_size / upscale - max(bbox_shape))) // 2 # do not upscale small bbox too much
            large_m = int(self.final_size / downscale - max(bbox_shape)) // 2 # do not downsample bbox too much
            if min(bbox_shape) > ignore_size and large_m > small_m:
                break
            else:
                large_m = small_m + 8
        for try_idx in range(200)[::-1]:
            crop_x1 = np.random.randint(min_x-large_m, min_x-small_m)
            crop_y1 = np.random.randint(min_y-large_m, min_y-small_m)
            if try_idx<100:
                crop_x1 = crop_y1 = 0
            cur_min_size = int(max(max_x + small_m - crop_x1, max_y + small_m - crop_y1))
            cur_max_size = int(max(bbox_shape) + 2*large_m)
            if cur_min_size < cur_max_size:
                target_size = np.random.randint(cur_min_size, cur_max_size)
                crop_x2 = int(crop_x1 + target_size)
                crop_y2 = int(crop_y1 + target_size)
                break
            if try_idx<1:
                print("try too much times for random crop", (min_x, min_y, max_x, max_y), (crop_x1, crop_y1), (small_m, large_m))
                crop_x1 = crop_y1 = 0
                crop_x2 = crop_y2 = target_size = max(img.shape[0], img.shape[1])
        debug_str = "origin: {}, bbox:{} , crop:{}, target_size {}, margin:{}".format(origin_shape,
                                                        (min_x, min_y, max_x, max_y),
                                                        (crop_x1, crop_y1, crop_x2, crop_y2), (bbox_shape, target_size),
                                                        (small_m, large_m))
        # pad the image
        pad_dims = ((max(-crop_y1, 0), max(crop_y2-img.shape[0], 0)),
                    (max(-crop_x1, 0), max(crop_x2-img.shape[1], 0)),
                    (0, 0))
        img = np.pad(img, pad_dims, 'constant', constant_values=0)
        y1 = max(crop_y1, 0)
        x1 = max(crop_x1, 0)
        img = img[y1:y1+target_size, x1:x1+target_size, :]
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape
        
        # post processing: gt_bboxes
        for key in ['gt_bboxes', 'gt_bboxes_ignore']:
            if key in results.get('bbox_fields', []):
                bboxes = results[key]
                bboxes[:, 0::2] -= float(crop_x1)
                bboxes[:, 1::2] -= float(crop_y1)
                results[key] = bboxes
        for key in ['gt_coefs']:
            if key in results.get('bbox_fields', []):
                bboxes = results[key]
                bboxes[:, -2] -= float(crop_x1)
                bboxes[:, -1] -= float(crop_y1)
                results[key] = bboxes
                
        # valid gt bboxes indices
        bboxes = results['gt_bboxes']
        min_x = np.min(bboxes[:, 0::2], axis=-1)
        min_y = np.min(bboxes[:, 1::2], axis=-1)
        max_x = np.max(bboxes[:, 0::2], axis=-1)
        max_y = np.max(bboxes[:, 1::2], axis=-1)
        bbox_size = np.minimum(max_x - min_x + 1.0, max_y - min_y + 1.0)
        valid_inds = (min_x>=0) & (min_y>=0) & (
                        max_x<img_shape[1]) & (
                        max_y<img_shape[0])
        valid_size = bbox_size/img_shape[0]*self.final_size>ignore_size # avoid too small bbox
        if np.sum(valid_inds & valid_size)>0:
            valid_inds = valid_inds & valid_size
        elif np.sum(valid_inds)==0:
            print("no valid bboxes {}\nnow bboxes: {}, size: {}, img_shape: {}".format(debug_str, bboxes, bbox_size, img_shape))
        # add non valid to ignore
        non_valid_bboxes = results['gt_bboxes'][~valid_inds, :]
        origin_ignores = results['gt_bboxes_ignore']
        results['gt_bboxes_ignore'] = np.concatenate((origin_ignores, non_valid_bboxes), axis=0)
        # re-assign by valid_inds, remove non valid bboxes
        for key in ['gt_bboxes', 'gt_skeleton', 'gt_coefs']:
            if key in results.get('bbox_fields', []):
                bboxes = results[key]
                results[key] = bboxes[valid_inds, :]
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(ratio_range={})'.format(
            self.ratio_range)
    
    
@PIPELINES.register_module
class CurveRandomRotate(object):
    """Random rotate the image & bboxes

    Args:
        limit_angle: 15, rotate from -15 to 15 degree
    """

    def __init__(self, limit_angle=15.0):
        assert limit_angle>=0 and limit_angle<180, "limit_angle should be an integer in [0, 180)"
        self.limit_angle = limit_angle

    def __call__(self, results):
        img = results['img']
        height, width, _ = img.shape
        # rotate matrix
        angle = random.uniform(-self.limit_angle, self.limit_angle)
        matrix = cv2.getRotationMatrix2D(center=(width*0.5, height*0.5), angle=angle, scale=1.0)
        # rotate image, first get outside points
        heightNew = int(width * math.fabs(math.sin(math.radians(angle))) + height * math.fabs(math.cos(math.radians(angle))))
        widthNew = int(height * math.fabs(math.sin(math.radians(angle))) + width * math.fabs(math.cos(math.radians(angle))))
        # add shift to avoid clipping
        matrix[0,2] += (widthNew - width)//2
        matrix[1,2] += (heightNew - height)//2
        # rotate cheby coefs and bboxes:
        valid_inds = None
        for key in ['gt_coefs']:
            bboxes = results[key].copy()
            centers = bboxes[:, -2:].reshape(bboxes.shape[0], -1, 2)
            centers = cv2.transform(centers, matrix)
            bboxes[:, -2:] = centers.reshape(bboxes.shape[0], -1)
            bboxes[:, -4] -= math.radians(angle)/math.pi
            valid_inds = (bboxes[:, -4]>=0) & (bboxes[:, -4]<1)
            if np.sum(valid_inds)==0:
                return results # if no valid bboxes, return original results
            results[key] = bboxes
        # rotate bboxes
        for key in ['gt_bboxes', 'gt_bboxes_ignore']:
            if key in results.get('bbox_fields', []):
                bboxes = results[key]
                if bboxes.shape[0]>0:
                    bboxes = bboxes.reshape(bboxes.shape[0], -1, 2)
                    bboxes = cv2.transform(bboxes, matrix)
                    bboxes = bboxes.reshape(bboxes.shape[0], -1)
                    results[key] = bboxes
        if valid_inds is not None:
            non_valid_bboxes = results['gt_bboxes'][~valid_inds, :]
            origin_ignores = results['gt_bboxes_ignore']
            results['gt_bboxes_ignore'] = np.concatenate((origin_ignores, non_valid_bboxes), axis=0)
            # re-assign by valid_inds, remove non valid bboxes
            for key in ['gt_bboxes', 'gt_skeleton', 'gt_coefs']:
                bboxes = results[key]
                results[key] = bboxes[valid_inds, :]
        # rotate image
        img = cv2.warpAffine(img, matrix, dsize=(widthNew, heightNew))
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(limit_angle={})'.format(
            self.limit_angle)

@PIPELINES.register_module
class CurveSegResizeFlipPadRescale(object):
    """A sequential transforms to semantic segmentation maps.

    The same pipeline as input images is applied to the semantic segmentation
    map, and finally rescale it by some scale factor. The transforms include:
    1. resize
    2. flip
    3. pad
    4. rescale (so that the final size can be different from the image size)

    Args:
        scale_factor (float): The scale factor of the final output.
    """

    def __init__(self, scale_factor=1):
        self.scale_factor = scale_factor

    def __call__(self, results):
        if results['keep_ratio']:
            gt_seg = mmcv.imrescale(
                results['gt_semantic_seg'],
                results['scale'],
                interpolation='nearest')
        else:
            gt_seg = mmcv.imresize(
                results['gt_semantic_seg'],
                results['scale'],
                interpolation='nearest')
        if results['flip']:
            gt_seg = mmcv.imflip(gt_seg)
        if gt_seg.shape != results['pad_shape']:
            gt_seg = mmcv.impad(gt_seg, results['pad_shape'][:2])
        if self.scale_factor != 1:
            gt_seg = mmcv.imrescale(
                gt_seg, self.scale_factor, interpolation='nearest')
        results['gt_semantic_seg'] = gt_seg
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(scale_factor={})'.format(
            self.scale_factor)

@PIPELINES.register_module
class CurveExpand(object):
    """Random expand the image & bboxes.

    Randomly place the original image on a canvas of 'ratio' x original image
    size filled with mean values. The ratio is in the range of ratio_range.

    Args:
        mean (tuple): mean value of dataset.
        to_rgb (bool): if need to convert the order of mean to align with RGB.
        ratio_range (tuple): range of expand ratio.
    """

    def __init__(self, mean=(0, 0, 0), to_rgb=True, ratio_range=(1, 4)):
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range

    def __call__(self, results):
        img, boxes = [results[k] for k in ('img', 'gt_bboxes')]

        h, w, c = img.shape
        results['img_shape'] = img.shape
        # square expand_img
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        new_size = int(max(h, w) * ratio)
        expand_img = np.full((new_size, new_size, c),
                             self.mean).astype(img.dtype)
        # place img on center of expand_img
        left = int((new_size - w)/2.0)
        top = int((new_size - h)/2.0)
        expand_img[top:top + h, left:left + w] = img
        results['img'] = expand_img
        # transform boxes

        for key in ['gt_bboxes', 'gt_bboxes_ignore']:
            if key in results.get('bbox_fields', []):
                bboxes = results[key]
                bboxes[:, 0::2] += float(left)
                bboxes[:, 1::2] += float(top)
                results[key] = bboxes
        for key in ['gt_coefs']:
            if key in results.get('bbox_fields', []):
                bboxes = results[key]
                bboxes[:, -2] += float(left)
                bboxes[:, -1] += float(top)
                results[key] = bboxes

        if 'gt_masks' in results:
            expand_gt_masks = []
            for mask in results['gt_masks']:
                expand_mask = np.full((int(h * ratio), int(w * ratio)),
                                      0).astype(mask.dtype)
                expand_mask[top:top + h, left:left + w] = mask
                expand_gt_masks.append(expand_mask)
            results['gt_masks'] = expand_gt_masks
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mean={}, to_rgb={}, ratio_range={})'.format(
            self.mean, self.to_rgb, self.ratio_range)
        return repr_str


@PIPELINES.register_module
class CurveMinIoURandomCrop(object):
    """Random crop the image & bboxes, the cropped patches have minimum IoU
    requirement with original image & bboxes, the IoU threshold is randomly
    selected from min_ious.

    Args:
        min_ious (tuple): minimum IoU threshold
        crop_size (tuple): Expected size after cropping, (h, w).
    """

    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        # 1: return ori img
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, results):
        img, boxes, labels = [
            results[k] for k in ('img', 'gt_bboxes', 'gt_labels')
        ]
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            if mode == 1:
                return results

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h)))
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = (center[:, 0] > patch[0]) * (
                    center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (
                        center[:, 1] < patch[3])
                if not mask.any():
                    continue
                boxes = boxes[mask]
                labels = labels[mask]

                # adjust boxes
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                boxes -= np.tile(patch[:2], 2)

                results['img'] = img
                results['gt_bboxes'] = boxes
                results['gt_labels'] = labels

                if 'gt_masks' in results:
                    valid_masks = [
                        results['gt_masks'][i] for i in range(len(mask))
                        if mask[i]
                    ]
                    results['gt_masks'] = [
                        gt_mask[patch[1]:patch[3], patch[0]:patch[2]]
                        for gt_mask in valid_masks
                    ]
                return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(min_ious={}, min_crop_size={})'.format(
            self.min_ious, self.min_crop_size)
        return repr_str
