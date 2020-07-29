import torch

from mmdet.core.bbox import PseudoSampler, build_assigner
from curve.core.bbox.assign_sampling import assign_and_sample
from mmdet.core.anchor.anchor_target import images_to_levels, anchor_inside_flags, unmap
from mmdet.core.utils import multi_apply
from ..bbox.transforms import bbox2radius


def radius_anchor_target(anchor_list,
                        valid_flag_list,
                        bbox_preds,
                        gt_bboxes_list,
                        gt_cheby_list,
                        gt_skeleton_list,
                        img_metas,
                        target_means,
                        target_stds,
                        num_coords,
                        cfg,
                        gt_bboxes_ignore_list=None,
                        gt_labels_list=None,
                        label_channels=1,
                        sampling=True,
                        unmap_outputs=True):
    num_imgs = len(img_metas)
    assert len(anchor_list) == len(valid_flag_list) == num_imgs

    # anchor number of multi levels
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    # concat all level anchors and flags to a single tensor
    for i in range(num_imgs):
        assert len(anchor_list[i]) == len(valid_flag_list[i])
        anchor_list[i] = torch.cat(anchor_list[i])
        valid_flag_list[i] = torch.cat(valid_flag_list[i])

    # compute targets for each image
    NoneList = lambda x: [None] * num_imgs if x is None else x
    gt_bboxes_ignore_list = NoneList(gt_bboxes_ignore_list)
    gt_labels_list = NoneList(gt_labels_list)
    gt_cheby_list = NoneList(gt_cheby_list)
    gt_skeleton_list = NoneList(gt_skeleton_list)

    bbox_pred_list = []
    for i in range(num_imgs):
        pred_list_per_img = []
        for bbox_pred in bbox_preds:
            bbox_pred = bbox_pred[i, ...].permute(1, 2, 0)
            bbox_pred = torch.flatten(bbox_pred, start_dim=0, end_dim=1)  # [h*w,  d]
            pred_list_per_img.append(bbox_pred)
        bbox_pred_list.append(torch.cat(pred_list_per_img))

    all_targets = multi_apply(
        radius_target_single,
        anchor_list,
        valid_flag_list,
        bbox_pred_list,
        gt_bboxes_list,
        gt_cheby_list,
        gt_skeleton_list,
        gt_bboxes_ignore_list,
        gt_labels_list,
        img_metas,
        target_means=target_means,
        target_stds=target_stds,
        num_coords=num_coords,
        cfg=cfg,
        label_channels=label_channels,
        sampling=sampling,
        unmap_outputs=unmap_outputs)

    (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
     all_ctr_targets, all_ctr_weights,
     pos_inds_list, neg_inds_list) = all_targets
    # no valid anchors
    if any([labels is None for labels in all_labels]):
        return None
    # print('all_bbox_targets', all_bbox_targets)
    # sampled anchors of all images
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    # split targets to a list w.r.t. multiple levels
    labels_list = images_to_levels(all_labels, num_level_anchors)
    label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
    bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
    bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
    ctr_targets_list = images_to_levels(all_ctr_targets, num_level_anchors)
    ctr_weights_list = images_to_levels(all_ctr_weights, num_level_anchors)
    return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
            ctr_targets_list, ctr_weights_list, num_total_pos, num_total_neg)


def radius_target_single(flat_anchors,
                        valid_flags,
                        bbox_pred,
                        gt_bboxes,
                        gt_cheby,
                        gt_skeleton,
                        gt_bboxes_ignore,
                        gt_labels,
                        img_meta,
                        target_means,
                        target_stds,
                        num_coords,
                        cfg,
                        label_channels=1,
                        sampling=True,
                        unmap_outputs=True):
    inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                       img_meta['img_shape'][:2],
                                       cfg.allowed_border)
    if not inside_flags.any():
        return (None, ) * 6
    # assign gt and sample anchors
    anchors = flat_anchors[inside_flags, :]
#     print('at cheby_target, gt_bboxes_ignore:', gt_bboxes_ignore)
    if sampling:
        assign_result, sampling_result = assign_and_sample(
            anchors, gt_bboxes, gt_cheby, gt_skeleton, gt_bboxes_ignore, None, cfg)
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        assign_result = bbox_assigner.assign(anchors, gt_bboxes,
                                             gt_bboxes_ignore, gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors,
                                              gt_bboxes)

    num_valid_anchors = anchors.shape[0]

    labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
    bbox_targets = torch.zeros((num_valid_anchors, num_coords-3), dtype=torch.float).cuda()
    bbox_weights = torch.zeros((num_valid_anchors, num_coords-3), dtype=torch.float).cuda()
    ctr_targets = torch.zeros((num_valid_anchors, 3), dtype=torch.float).cuda()
    ctr_weights = torch.zeros((num_valid_anchors, 3), dtype=torch.float).cuda()

    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        deltas, weights = bbox2radius(sampling_result.pos_bboxes,
                                     sampling_result.pos_gt_bboxes,
                                     sampling_result.pos_gt_skeleton, num_coords, 
                                     target_means, target_stds)
        
        bbox_targets[pos_inds, :] = deltas[:, :-3]
        bbox_weights[pos_inds, :] = weights.unsqueeze(1) if cfg.use_centerness else 1.0
        ctr_targets[pos_inds, :] = deltas[:, -3:]
        ctr_weights[pos_inds, :] = weights.unsqueeze(1) if cfg.use_centerness else 1.0
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = weights if cfg.use_centerness else 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
        # print("pos:", len(pos_inds), "neg:", len(neg_inds),  weights)
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

    # map up to original set of anchors
    if unmap_outputs:
        num_total_anchors = flat_anchors.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
        ctr_targets = unmap(ctr_targets, num_total_anchors, inside_flags)
        ctr_weights = unmap(ctr_weights, num_total_anchors, inside_flags)
    return (labels, label_weights, bbox_targets, bbox_weights, ctr_targets, ctr_weights, pos_inds,
            neg_inds)
