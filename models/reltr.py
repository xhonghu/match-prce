# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.
import torch
import torch.nn.functional as F
from torch import nn
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer


class RelTR(nn.Module):
    """
    RelTR: 关系 Transformer for Scene Graph Generation
    """

    def __init__(self, backbone, transformer, num_classes, num_entities, num_triplets, aux_loss=False):
        super().__init__()
        # 用于目标检测查询数量  100
        self.num_entities = num_entities
        # transformer 输入输出维度 256
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        # 它将输入的特征图（维度为 backbone.num_channels 2048）经过卷积操作后转成transformer可以输入的维度 256
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        # 图片输入经过backone转成2048的特征图
        self.backbone = backbone
        # 开启auxloss，每层Transformer都输出loss结果，共6层
        self.aux_loss = aux_loss
        # 实体Embedding [100,256 * 2]
        self.entity_embed = nn.Embedding(num_entities, hidden_dim * 2)
        # 三元组查询 [200,256 * 3]
        self.triplet_embed = nn.Embedding(num_triplets, hidden_dim * 3)
        self.so_embed = nn.Embedding(2, hidden_dim)  # subject and object encoding

        # entity prediction
        # (entity_class_embed): Linear(in_features=256, out_features=152)
        self.entity_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        # (entity_bbox_embed): Linear(in_features=256, out_features=4)
        self.entity_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # subject/object label classfication and box regression
        # (sub_class_embed): Linear(in_features=256, out_features=152)
        self.sub_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        # (sub_bbox_embed): MLP(in_features=256, out_features=4)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # (obj_class_embed): Linear(in_features=256, out_features=152)
        self.obj_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        # (obj_bbox_embed): MLP(in_features=256, out_features=4)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, samples: NestedTensor):
        """
        The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the entity classification logits (including no-object) for all entity queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": the normalized entity boxes coordinates for all entity queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "sub_logits": the subject classification logits
               - "obj_logits": the object classification logits
               - "sub_boxes": the normalized subject boxes coordinates
               - "obj_boxes": the normalized object boxes coordinates
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        # 从图片中提取特征,以及位置编码。
        features, pos = self.backbone(samples)
        # 从特征张量 features 中获取最后一层的特征表示，并使用 decompose 方法将其分解为源张量 src 和掩码张量 mask。
        src, mask = features[-1].decompose()
        h, w = mask.shape[-2], mask.shape[-1]
        assert mask is not None

        # hs 是pre_logits的原始输出用来做目标检测的 [6,1,100,256]
        # hs_t 就是模型得出的subject/object 查询对 [6,1,200,512]
        hs, hs_t, so_masks = self.transformer(self.input_proj(src), mask, self.entity_embed.weight,
                                              self.triplet_embed.weight, pos[-1], self.so_embed.weight)
        # 将hs_t分解开
        hs_sub, hs_obj = torch.split(hs_t, self.hidden_dim, dim=-1)

        outputs_class = self.entity_class_embed(hs)  # [6, bs, 100, 152]
        outputs_coord = self.entity_bbox_embed(hs).sigmoid()  # [6,bs,100, 4]

        outputs_class_sub = self.sub_class_embed(hs_sub)  # [6, bs, 200, 152]
        outputs_coord_sub = self.sub_bbox_embed(hs_sub).sigmoid()  # [6, bs, 200, 4]

        outputs_class_obj = self.obj_class_embed(hs_obj)  # [6, bs, 200, 152]
        outputs_coord_obj = self.obj_bbox_embed(hs_obj).sigmoid()  # [6, bs, 200, 4]

        so_masks = so_masks.view(-1, 2, h, w)

        # 这里取了最后一层的解码结果
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
               'sub_logits': outputs_class_sub[-1], 'sub_boxes': outputs_coord_sub[-1],
               'obj_logits': outputs_class_obj[-1], 'obj_boxes': outputs_coord_obj[-1]
               }
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_class_sub, outputs_coord_sub,
                                                    outputs_class_obj, outputs_coord_obj)
        return out, hs_t, so_masks

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_class_sub, outputs_coord_sub,
                      outputs_class_obj, outputs_coord_obj):
        return [{'pred_logits': a, 'pred_boxes': b, 'sub_logits': c, 'sub_boxes': d, 'obj_logits': e, 'obj_boxes': f}
                for a, b, c, d, e, f in zip(outputs_class[:-1], outputs_coord[:-1], outputs_class_sub[:-1],
                                            outputs_coord_sub[:-1], outputs_class_obj[:-1], outputs_coord_obj[:-1])]


class MLP(nn.Module):
    """
    Very simple multi-layer perceptron (also called FFN)
    非常简单的多层感知机,也叫做FFN,用于回归任务
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SetCriterion(nn.Module):
    """
    这个类是用来计算损失的
    The process happens in two steps:
        1) 我们计算了真值框和模型输出之间的匈牙利分配
        2) 我们监督每对匹配的标注/预测（监督类和框）
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """
        Create the criterion.
        Parameters:
            num_classes: 对象类别数，省略特殊的无对象类别
            matcher: 能够计算目标和计算结果之间的匹配度的模块
            weight_dict: dict 包含损失的名称作为键，并作为值包含它们的相对权重。
            eos_coef: 应用于无对象类别的相对分类权重
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """
        类别的损失
        """
        assert 'pred_logits' in outputs

        pred_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices[0])
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices[0])])
        target_classes = torch.full(pred_logits.shape[:2], self.num_classes, dtype=torch.int64,
                                    device=pred_logits.device)
        target_classes[idx] = target_classes_o

        sub_logits = outputs['sub_logits']
        obj_logits = outputs['obj_logits']
        rel_idx = self._get_src_permutation_idx(indices[1])
        target_rels_classes_o = torch.cat(
            [t["labels"][t["rel_annotations"][J, 0]] for t, (_, J) in zip(targets, indices[1])])
        target_relo_classes_o = torch.cat(
            [t["labels"][t["rel_annotations"][J, 1]] for t, (_, J) in zip(targets, indices[1])])
        target_sub_classes = torch.full(sub_logits.shape[:2], self.num_classes, dtype=torch.int64,
                                        device=sub_logits.device)
        target_obj_classes = torch.full(obj_logits.shape[:2], self.num_classes, dtype=torch.int64,
                                        device=obj_logits.device)
        target_sub_classes[rel_idx] = target_rels_classes_o
        target_obj_classes[rel_idx] = target_relo_classes_o
        # 这里将三个损失拼接
        target_classes = torch.cat((target_classes, target_sub_classes, target_obj_classes), dim=1)
        src_logits = torch.cat((pred_logits, sub_logits, obj_logits), dim=1)
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction='none')
        losses = {'loss_ce': loss_ce.sum() / self.empty_weight[target_classes].sum()}
        # 在log中添加预测的错误率
        if log:
            losses['class_error'] = 100 - accuracy(pred_logits[idx], target_classes_o)[0]
            losses['sub_error'] = 100 - accuracy(sub_logits[rel_idx], target_rels_classes_o)[0]
            losses['obj_error'] = 100 - accuracy(obj_logits[rel_idx], target_relo_classes_o)[0]
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        计算目标框的损失
        the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h)
        通过图片的大小归一化
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices[0])
        pred_boxes = outputs['pred_boxes'][idx]
        target_entry_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices[0])], dim=0)

        rel_idx = self._get_src_permutation_idx(indices[1])
        target_rels_boxes = torch.cat(
            [t['boxes'][t["rel_annotations"][i, 0]] for t, (_, i) in zip(targets, indices[1])], dim=0)
        target_relo_boxes = torch.cat(
            [t['boxes'][t["rel_annotations"][i, 1]] for t, (_, i) in zip(targets, indices[1])], dim=0)
        rels_boxes = outputs['sub_boxes'][rel_idx]
        relo_boxes = outputs['obj_boxes'][rel_idx]

        src_boxes = torch.cat((pred_boxes, rels_boxes, relo_boxes), dim=0)
        target_boxes = torch.cat((target_entry_boxes, target_rels_boxes, target_relo_boxes), dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """
        This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        self.indices = indices
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) + len(t["rel_annotations"]) * 2 for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels' or loss == 'relations':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ 这个模型用来将模型输出，转化成coco api期望的格式，评估的时候用的"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """
        Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """

        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        sub_logits, sub_bbox = outputs['sub_logits'], outputs['sub_boxes']
        obj_logits, obj_bbox = outputs['obj_logits'], outputs['obj_boxes']

        assert len(out_logits) == len(target_sizes)
        assert len(sub_logits) == len(target_sizes)
        assert len(obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        sub_prob = F.softmax(sub_logits, -1)
        sub_scores, sub_labels = sub_prob[..., :-1].max(-1)

        obj_prob = F.softmax(obj_logits, -1)
        obj_scores, obj_labels = obj_prob[..., :-1].max(-1)


        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        sub_boxes = box_ops.box_cxcywh_to_xyxy(sub_bbox)
        obj_boxes = box_ops.box_cxcywh_to_xyxy(obj_bbox)

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

        boxes = boxes * scale_fct[:, None, :]
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        sub_results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(sub_scores, sub_labels, sub_boxes)]
        obj_results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(obj_scores, obj_labels, obj_boxes)]

        return results, sub_results, obj_results


def build(args):
    # some entity categories in OIV6 are deactivated.
    num_classes = 151 if args.dataset != 'oi' else 289
    device = torch.device(args.device)
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    ##用于计算预测三元组和真值三元组之间的最优目标匹配（匈牙利匹配算法）
    matcher = build_matcher(args)
    ##初始化模型
    model = RelTR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_entities=args.num_entities,
        num_triplets=args.num_triplets,
        aux_loss=args.aux_loss)

    # 这里是loss里各个权重的定义
    weight_dict = {'loss_ce': 1}
    weight_dict['loss_bbox'] = args.bbox_loss_coef
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    losses = ['labels', 'boxes']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    # 这里是坐标框的后处理，评估的时候用的。
    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors
