import argparse
import random
from pprint import pprint
from torch.utils.data import DataLoader
from datasets import build_dataset
from rel_class import build_rel_class
import numpy as np
import torch
import copy
import util.misc as utils
from util.misc import nested_tensor_from_tensor_list
from util.box_ops import rescale_bboxes
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, calculate_mR_from_evaluator_list


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int)

    # dataset parameters
    parser.add_argument('--dataset', default='vg')
    parser.add_argument('--ann_path', default='./data/vg/vg_init/', type=str)
    parser.add_argument('--img_folder', default='./data/vg/images/', type=str)
    parser.add_argument('--output_dir', default='./output/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--eval', action='store_false')
    parser.add_argument('--num_workers', default=2, type=int)
    return parser


def main(args):
    utils.init_distributed_mode(args)
    pprint(vars(args))
    print("***************************************")
    print()
    device = torch.device(args.device)
    # fix the seed for reproducibility
    # 固定种子,实现实验的可重复性
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    match, model, criterion = build_rel_class(args)
    model1 = copy.deepcopy(model)
    model2 = copy.deepcopy(model)
    model3 = copy.deepcopy(model)
    model4 = copy.deepcopy(model)
    model5 = copy.deepcopy(model)
    model1.to(device)
    model2.to(device)
    model3.to(device)
    model4.to(device)
    model5.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 + sum(
        p.numel() for p in match.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    dataset_val = build_dataset(image_set='val', args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # 加载各个分类器权重
    checkpoint1 = torch.load('./exper/trainset2/checkpoint0049.pth')['model']
    checkpoint2 = torch.load('./exper/trainset3/checkpoint0049.pth')['model']
    checkpoint3 = torch.load('./exper/trainset4/checkpoint0049.pth')['model']
    checkpoint4 = torch.load('./exper/trainset5/checkpoint0049.pth')['model']
    checkpoint5 = torch.load('./exper/sample/train5/checkpoint0029.pth')['model']
    model1.load_state_dict(checkpoint1, strict=True)
    model2.load_state_dict(checkpoint2, strict=True)
    model3.load_state_dict(checkpoint3, strict=True)
    model4.load_state_dict(checkpoint4, strict=True)
    model5.load_state_dict(checkpoint5, strict=True)
    print('开始在测试集上测试模型...........')
    test_stats = evaluate(match, model1, model2, model3, model4, model5, criterion, data_loader_val, device)


@torch.no_grad()
def evaluate(match, model1, model2, model3, model4, model5, criterion, data_loader, device):
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
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

        outputs1 = model1(outputs, hs_t, so_masks).copy()
        outputs2 = model2(outputs, hs_t, so_masks).copy()
        outputs3 = model3(outputs, hs_t, so_masks).copy()
        outputs4 = model4(outputs, hs_t, so_masks).copy()
        outputs5 = model5(outputs, hs_t, so_masks).copy()

        loss_dict = criterion(outputs, targets)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_value = loss_dict_reduced['loss_rel']
        metric_logger.update(loss=loss_value)
        metric_logger.update(sub_error=loss_dict_reduced['sub_error'])
        metric_logger.update(obj_error=loss_dict_reduced['obj_error'])
        metric_logger.update(rel_error=loss_dict_reduced['rel_error'])

        evaluate_rel_batch(outputs1, outputs2, outputs3, outputs4, outputs5, targets, evaluator, evaluator_list)

    evaluator['sgdet'].print_stats()
    calculate_mR_from_evaluator_list(evaluator_list, 'sgdet')
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate_rel_batch(outputs1, outputs2, outputs3, outputs4, outputs5, targets, evaluator, evaluator_list):
    for batch, target in enumerate(targets):
        # recovered boxes with original size
        target_bboxes_scaled = rescale_bboxes(target['boxes'].cpu(),
                                              torch.flip(target['orig_size'], dims=[0]).cpu()).clone().numpy()

        gt_entry = {'gt_classes': target['labels'].cpu().clone().numpy(),
                    'gt_relations': target['rel_annotations'].cpu().clone().numpy(),
                    'gt_boxes': target_bboxes_scaled}

        sub_bboxes_scaled = rescale_bboxes(outputs1['sub_boxes'][batch].cpu(),
                                           torch.flip(target['orig_size'], dims=[0]).cpu()).clone().numpy()
        obj_bboxes_scaled = rescale_bboxes(outputs1['obj_boxes'][batch].cpu(),
                                           torch.flip(target['orig_size'], dims=[0]).cpu()).clone().numpy()

        pred_sub_scores, pred_sub_classes = torch.max(outputs1['sub_logits'][batch].softmax(-1)[:, :-1], dim=1)
        pred_obj_scores, pred_obj_classes = torch.max(outputs1['obj_logits'][batch].softmax(-1)[:, :-1], dim=1)

        # 训练的各个分类器在这一步综合打分
        # ###################################################################
        rel_scores1 = outputs1['rel_logits'][batch][:, 1:-1].softmax(-1).cpu()  
        rel_scores2 = outputs2['rel_logits'][batch][:, 1:-1].softmax(-1).cpu()
        rel_scores3 = outputs3['rel_logits'][batch][:, 1:-1].softmax(-1).cpu()
        rel_scores4 = outputs4['rel_logits'][batch][:, 1:-1].softmax(-1).cpu()
        rel_scores5 = outputs5['rel_logits'][batch][:, 1:-1].softmax(-1).cpu()
        rel_scores = []
        for i in range(0, 200):
            rel_list = [rel_scores1[i], rel_scores2[i], rel_scores3[i], rel_scores4[i], rel_scores5[i]]
            rel_index = [rel_scores1[i].argmax(), rel_scores2[i].argmax(), rel_scores3[i].argmax(), rel_scores4[i].argmax(),
                         rel_scores5[i].argmax()]
            rel_score = [rel_scores1[i].max(), rel_scores2[i].max(), rel_scores3[i].max(), rel_scores4[i].max(),
                         rel_scores5[i].max()]

            # 计算一维数组的众数及其下标
            unique, counts = np.unique(rel_index, return_counts=True)
            max_count = np.max(counts)
            mode_indices = np.where(counts == max_count)[0]
            modes = unique[mode_indices]
            mode_indices = np.where(np.isin(rel_index, modes))[0]

            # 取众数中对应得分最高的下标
            index = rel_score.index(max(rel_score[i] for i in mode_indices))
            rel_scores.append(rel_list[index].tolist())
            
            # index = rel_score.index(max(rel_score[i] for i in range(5)))
            # rel_scores.append(rel_list[index].tolist())
        rel_scores = torch.Tensor(rel_scores)

        # ###################################################################A-relation-A
        # Remove some useless pairs like A-relation-A
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RelTR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
