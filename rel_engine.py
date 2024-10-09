# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.

"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable
import numpy as np
import torch
import copy
import util.misc as utils
from util.misc import nested_tensor_from_tensor_list
from util.box_ops import rescale_bboxes
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, calculate_mR_from_evaluator_list


def train_one_epoch(match: torch.nn.Module, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('sub_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('obj_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('rel_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 500

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs, hs_t, so_masks = match(samples)
        hs_t.to(device)
        so_masks.to(device)
        outputs = model(outputs, hs_t, so_masks)
        loss_dict = criterion(outputs, targets)
        losses = loss_dict['loss_rel']
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_value = loss_dict_reduced['loss_rel']

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value)
        metric_logger.update(sub_error=loss_dict_reduced['sub_error'])
        metric_logger.update(obj_error=loss_dict_reduced['obj_error'])
        metric_logger.update(rel_error=loss_dict_reduced['rel_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(match, model, criterion, data_loader, device):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('sub_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('obj_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('rel_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=False)
    evaluator_list = []
    for index, name in enumerate(data_loader.dataset.rel_categories):
        if index == 0:
            continue
        evaluator_list.append((index, name, BasicSceneGraphEvaluator.all_modes()))

    for samples, targets in metric_logger.log_every(data_loader, 500, header):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs, hs_t, so_masks = match(samples)
        hs_t.to(device)
        so_masks.to(device)

        outputs = model(outputs, hs_t, so_masks)

        loss_dict = criterion(outputs, targets)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_value = loss_dict_reduced['loss_rel']
        metric_logger.update(loss=loss_value)
        metric_logger.update(sub_error=loss_dict_reduced['sub_error'])
        metric_logger.update(obj_error=loss_dict_reduced['obj_error'])
        metric_logger.update(rel_error=loss_dict_reduced['rel_error'])

        evaluate_rel_batch(outputs, targets, evaluator, evaluator_list)

    evaluator['sgdet'].print_stats()
    calculate_mR_from_evaluator_list(evaluator_list, 'sgdet')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate_rel_batch(outputs, targets, evaluator, evaluator_list):
    for batch, target in enumerate(targets):
        # recovered boxes with original size
        target_bboxes_scaled = rescale_bboxes(target['boxes'].cpu(),
                                              torch.flip(target['orig_size'], dims=[0]).cpu()).clone().numpy()

        gt_entry = {'gt_classes': target['labels'].cpu().clone().numpy(),
                    'gt_relations': target['rel_annotations'].cpu().clone().numpy(),
                    'gt_boxes': target_bboxes_scaled}

        sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][batch].cpu(),
                                           torch.flip(target['orig_size'], dims=[0]).cpu()).clone().numpy()
        obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][batch].cpu(),
                                           torch.flip(target['orig_size'], dims=[0]).cpu()).clone().numpy()

        pred_sub_scores, pred_sub_classes = torch.max(outputs['sub_logits'][batch].softmax(-1)[:, :-1], dim=1)
        pred_obj_scores, pred_obj_classes = torch.max(outputs['obj_logits'][batch].softmax(-1)[:, :-1], dim=1)

        # 训练的各个分类器再这一步综合打分
        rel_scores = outputs['rel_logits'][batch][:, 1:-1]  # 原论文
        rel_scores = rel_scores.softmax(-1)

        # ###################################################################A-relation-A
        # Remove some useless pairs like A-relation-A
        # # mask = torch.logical_and((pred_sub_classes - pred_obj_classes != 0).cpu(), torch.logical_and(pred_obj_scores >= 0.002, pred_sub_scores >= 0.002).cpu())
        mask = (pred_sub_classes - pred_obj_classes != 0).cpu()
        if mask.sum() < 200:
            sub_bboxes_scaled = sub_bboxes_scaled[mask]
            pred_sub_classes = pred_sub_classes[mask]
            pred_sub_scores = pred_sub_scores[mask]
            obj_bboxes_scaled = obj_bboxes_scaled[mask]
            pred_obj_classes = pred_obj_classes[mask]
            pred_obj_scores = pred_obj_scores[mask]
            rel_scores = rel_scores[mask]

            padded_indices = (pred_sub_scores + pred_obj_scores).sort(descending=True)[1][
                             : mask.shape[0] - mask.sum()].cpu()
            padded_sub_bboxes = sub_bboxes_scaled[padded_indices]
            if len(padded_indices) == 1:
                padded_sub_bboxes = np.expand_dims(padded_sub_bboxes, axis=0)
            padded_sub_class = pred_sub_classes[padded_indices]
            padded_sub_scores = pred_sub_scores[padded_indices]
            padded_obj_bboxes = obj_bboxes_scaled[padded_indices]
            if len(padded_indices) == 1:
                padded_obj_bboxes = np.expand_dims(padded_obj_bboxes, axis=0)
            padded_obj_class = pred_obj_classes[padded_indices]
            padded_obj_scores = pred_obj_scores[padded_indices]
            padded_rel_scores = rel_scores[padded_indices]
            max_value_indices = torch.max(padded_rel_scores, dim=1)[1]
            for i, idx in enumerate(max_value_indices):
                second_max_index = (-padded_rel_scores[i]).sort()[1][1]
                padded_rel_scores[i, second_max_index] += padded_rel_scores[i, idx] * 0.2
                padded_rel_scores[i, idx] = 0

            sub_bboxes_scaled = np.concatenate([sub_bboxes_scaled, padded_sub_bboxes], axis=0)
            pred_sub_classes = torch.cat([pred_sub_classes, padded_sub_class], dim=0)
            pred_sub_scores = torch.cat([pred_sub_scores, padded_sub_scores], dim=0)
            obj_bboxes_scaled = np.concatenate([obj_bboxes_scaled, padded_obj_bboxes], axis=0)
            pred_obj_classes = torch.cat([pred_obj_classes, padded_obj_class], dim=0)
            pred_obj_scores = torch.cat([pred_obj_scores, padded_obj_scores], dim=0)
            rel_scores = torch.cat([rel_scores, padded_rel_scores], dim=0)
        # ###################################################################A-relation-A

        pred_entry = {'sub_boxes': sub_bboxes_scaled,
                      'sub_classes': pred_sub_classes.cpu().clone().numpy(),
                      'sub_scores': pred_sub_scores.cpu().clone().numpy(),
                      'obj_boxes': obj_bboxes_scaled,
                      'obj_classes': pred_obj_classes.cpu().clone().numpy(),
                      'obj_scores': pred_obj_scores.cpu().clone().numpy(),
                      'rel_scores': rel_scores.cpu().clone().numpy()}

        evaluator['sgdet'].evaluate_scene_graph_entry(gt_entry, pred_entry)

        if evaluator_list is not None:
            for pred_id, _, evaluator_rel in evaluator_list:
                gt_entry_rel = gt_entry.copy()
                mask = np.in1d(gt_entry_rel['gt_relations'][:, -1], pred_id)
                gt_entry_rel['gt_relations'] = gt_entry_rel['gt_relations'][mask, :]
                if gt_entry_rel['gt_relations'].shape[0] == 0:
                    continue
                evaluator_rel['sgdet'].evaluate_scene_graph_entry(gt_entry_rel, pred_entry)
