# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.

"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable
import torch
from datasets.coco_eval import CocoEvaluator
import util.misc as utils

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('sub_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('obj_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    # 这里指定多少批次打印一次loss
    print_freq = 500
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs, hs_t, so_masks = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        # 规约之后的loss
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        # 每个loss按权重计算
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        # 如果loss为nan则停止训练
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # log打印的内容在这里改
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(sub_error=loss_dict_reduced['sub_error'])
        metric_logger.update(obj_error=loss_dict_reduced['obj_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # 汇聚每个进程的状态
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    #返回训练的状态
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('sub_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('obj_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    sub_coco_evaluator = CocoEvaluator(base_ds, iou_types)
    obj_coco_evaluator = CocoEvaluator(base_ds, iou_types)
    for samples, targets in metric_logger.log_every(data_loader, 200, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs, hs_t, so_masks = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(sub_error=loss_dict_reduced['sub_error'])
        metric_logger.update(obj_error=loss_dict_reduced['obj_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results, sub_results, obj_results = postprocessors['bbox'](outputs, orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        sub_res = {target['image_id'].item(): output for target, output in zip(targets, sub_results)}
        obj_res = {target['image_id'].item(): output for target, output in zip(targets, obj_results)}

        coco_evaluator.update(res)
        sub_coco_evaluator.update(sub_res)
        obj_coco_evaluator.update(obj_res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()
    sub_coco_evaluator.synchronize_between_processes()
    obj_coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    sub_coco_evaluator.accumulate()
    sub_coco_evaluator.summarize()
    obj_coco_evaluator.accumulate()
    obj_coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if 'bbox' in postprocessors.keys():
        stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        stats['sub_coco_eval_bbox'] = sub_coco_evaluator.coco_eval['bbox'].stats.tolist()
        stats['obj_coco_eval_bbox'] = obj_coco_evaluator.coco_eval['bbox'].stats.tolist()
    return stats, coco_evaluator
