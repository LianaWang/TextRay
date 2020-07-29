import json
import os
import os.path as osp

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import scatter, collate
from shapely.geometry import Polygon
from tqdm import tqdm

from mmdet.core.evaluation.eval_hooks import DistEvalHook


class TextDistEvalHook(DistEvalHook):
    def __init__(self, gt_json, score_thr=0.6, **kwargs):
        self.gt_json = TextDistEvalHook.list_to_dict(json.load(open(gt_json)))
        self.score_thr = score_thr
        super(TextDistEvalHook, self).__init__(**kwargs)

    @staticmethod
    def polygon_iou(poly1, poly2):
        if isinstance(poly1, list):
            poly1 = Polygon(np.array(poly1).reshape([-1, 2]))
        if isinstance(poly2, list):
            poly2 = Polygon(np.array(poly2).reshape([-1, 2]))
        iou = 0.0
        if poly1.intersects(poly2):
            inter_area = poly1.intersection(poly2).area
            union_area = poly1.area + poly2.area - inter_area
            iou = float(inter_area) / union_area
        return iou

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        results = [None for _ in range(len(self.dataset))]
        if runner.rank == 0:
            prog_bar = mmcv.ProgressBar(len(self.dataset))
        for idx in range(runner.rank, len(self.dataset), runner.world_size):
            data = self.dataset[idx]
            data_gpu = scatter(
                collate([data], samples_per_gpu=1),
                [torch.cuda.current_device()])[0]

            # compute output
            with torch.no_grad():
                result = runner.model(
                    return_loss=False, rescale=True, **data_gpu)
            # result is N polygons [#N, 72d + 1d]
            _bbox_lists = []
            for j in range(result.shape[0]):
                _bbox_lists.append(dict(
                    points=result[j, :-1].reshape((-1, 2)).tolist(),
                    score=result[j, -1])
                )
            results[idx] = dict(
                image_id=self.dataset.img_ids[idx],
                bboxes=_bbox_lists
            )
            batch_size = runner.world_size
            if runner.rank == 0:
                for _ in range(batch_size):
                    prog_bar.update()

        if runner.rank == 0:
            dist.barrier()
            for i in range(1, runner.world_size):
                tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(i))
                tmp_reads = mmcv.load(tmp_file)
                tmp_results = tmp_reads['results']
                for idx in range(i, len(results), runner.world_size):
                    results[idx] = tmp_results[idx]
                os.remove(tmp_file)
            self.evaluate(runner, results)
        else:
            tmp_file = osp.join(runner.work_dir,
                                'temp_{}.pkl'.format(runner.rank))
            writes = {
                'results': results
            }
            mmcv.dump(writes, tmp_file)
            dist.barrier()
        dist.barrier()

    @staticmethod
    def list_to_dict(list_results):
        _dict = {}
        for result in list_results:
            key = result.pop('image_id')
            _dict[key] = result
        return _dict

    def evaluate(self, runner, results):
        # evaluate jsons
        num_gt = 0
        num_pred = 0
        num_gt_correct = 0
        num_pred_correct = 0
        pred_json = TextDistEvalHook.list_to_dict(results)

        for img_id in tqdm(list(self.gt_json.keys())):
            pred_anns = pred_json[img_id]['bboxes']
            gt_anns = self.gt_json[img_id]['bboxes']
            # filter score thresholds
            pred_polygons = [_['points'] for _ in pred_anns if _['score'] > self.score_thr]
            # select gt_jsons
            gt_polygons = [_['points'] for _ in gt_anns if _['hard'] != 1]

            num_gt += len(gt_polygons)
            num_pred += len(pred_polygons)
            for gt_polygon in gt_polygons:
                gt_correct = False
                for pred_polygon in pred_polygons:
                    iou = TextDistEvalHook.polygon_iou(gt_polygon, pred_polygon)
                    if iou >= 0.5:
                        num_pred_correct += 1
                        gt_correct = True
                num_gt_correct += 1 if gt_correct else 0

        p = 1.0 * num_pred_correct / max(num_pred, 1e-10)
        r = 1.0 * num_gt_correct / max(num_gt, 1e-10)
        f = 2.0 * p * r / max(p + r, 1e-10)
        output_str = "Precision {:.3f}%, Recall {:.3f}%, F-measure {:.3f}%".format(p * 100.0, r * 100.0, f * 100.0)
        # print(output_str)
        # plot on tensor board
        runner.log_buffer.output['PRECISION'] = p * 100.0
        runner.log_buffer.output['RECALL'] = r * 100.0
        runner.log_buffer.output['F-Measure'] = f * 100.0
        runner.log_buffer.output['eval_result'] = '\n' + output_str + '\n'
        runner.log_buffer.ready = True


if __name__ == '__main__':
    from curve.datasets.CTW1500 import CTW1500
    import pickle

    data_root = '/home/chenyifeng/curve-text/data/TotalText/'
    hook = TextDistEvalHook(gt_json=data_root + 'total_test_gt.json', score_thr=0.999,
                            dataset=CTW1500(
                               ann_file=data_root + 'ImageSets/Main/train.txt',
                               img_prefix=data_root,
                               cache_root='Cache',
                               pipeline=[]
                           ))
    for img_id in tqdm(list(hook.gt_json.keys())):

        gt_anns = hook.gt_json[img_id]['bboxes']
        try:
            gt_polygons = [_['points'] for _ in gt_anns if _['hard'] != 1]
        except:
            print(gt_anns)
            break
    # _results = pickle.load(open('/tmp/ctw_test_pred2.pkl', 'rb'), encoding='latin1')
    # _image_ids = mmcv.list_from_file('/home/chenyifeng/curve-text/data/CTW1500/ImageSets/Main/test.txt')
    # # rearrange it into corresponding formats
    # results = []
    # for _id, result in zip(_image_ids, _results):
    #     _bbox_lists = []
    #     for j in range(result.shape[0]):
    #         _bbox_lists.append(dict(
    #             points=result[j, :-1].reshape((-1, 2)).tolist(),
    #             score=result[j, -1])
    #         )
    #     results.append(dict(
    #         image_id=_id,
    #         bboxes=_bbox_lists
    #     ))

    # hook.evaluate(None, results)
