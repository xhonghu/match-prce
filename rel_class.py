from models.backbone import Backbone, Joiner
from models.position_encoding import PositionEmbeddingSine
from models.transformer import Transformer
from models.reltr import RelTR
import torch
import torch.nn.functional as F
from torch import nn
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from rel_matcher import build_matcher


class RelClass(nn.Module):
    """
    RelTR: 关系 Transformer for Scene Graph Generation
    """

    def __init__(self, hidden_dim, num_rel_classes):
        super().__init__()
        # mask head
        self.hidden_dim = hidden_dim
        self.num_rel_classes = num_rel_classes
        self.so_mask_conv = nn.Sequential(torch.nn.Upsample(size=(28, 28)),
                                          nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=3, bias=True),
                                          nn.ReLU(inplace=True),
                                          nn.BatchNorm2d(64),
                                          nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                          nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True),
                                          nn.ReLU(inplace=True),
                                          nn.BatchNorm2d(32))
        self.so_mask_fc = nn.Sequential(nn.Linear(2048, 512),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(512, 128))


        # predicate classification
        self.rel_class_embed = MLP(hidden_dim * 2 + 128, hidden_dim, num_rel_classes + 1, 2)

    def forward(self, outputs, hs_t, so_masks):
        bs = hs_t.shape[1]
        so_masks = self.so_mask_conv(so_masks).view(6, bs, 200, -1)
        so_masks = self.so_mask_fc(so_masks)
        hs_sub, hs_obj = torch.split(hs_t, self.hidden_dim, dim=-1)
        outputs_class_rel = self.rel_class_embed(torch.cat((hs_sub, hs_obj, so_masks), dim=-1))
        outputs['rel_logits'] = outputs_class_rel[-1]
        return outputs


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class SetCriterion(nn.Module):
    def __init__(self, num_classes, num_rel_classes, matcher, eos_coef, losses):
        super().__init__()
        self.matcher = matcher
        self.losses = losses
        self.num_classes = num_classes
        self.num_rel_classes = num_rel_classes
        empty_weight_rel = torch.ones(num_rel_classes + 1)
        empty_weight_rel[-1] = eos_coef
        self.register_buffer('empty_weight_rel', empty_weight_rel)

    def loss_labels(self, outputs, targets, num_boxes, indices, log=True):
        sub_logits = outputs['sub_logits']
        obj_logits = outputs['obj_logits']
        rel_idx = self._get_src_permutation_idx(indices)
        target_rels_classes_o = torch.cat(
            [t["labels"][t["rel_annotations"][J, 0]] for t, (_, J) in zip(targets, indices)])
        target_relo_classes_o = torch.cat(
            [t["labels"][t["rel_annotations"][J, 1]] for t, (_, J) in zip(targets, indices)])

        target_sub_classes = torch.full(sub_logits.shape[:2], self.num_classes, dtype=torch.int64,
                                        device=sub_logits.device)
        target_obj_classes = torch.full(obj_logits.shape[:2], self.num_classes, dtype=torch.int64,
                                        device=obj_logits.device)

        target_sub_classes[rel_idx] = target_rels_classes_o
        target_obj_classes[rel_idx] = target_relo_classes_o
        losses = {}
        if log:
            losses['sub_error'] = 100 - accuracy(sub_logits[rel_idx], target_rels_classes_o)[0]
            losses['obj_error'] = 100 - accuracy(obj_logits[rel_idx], target_relo_classes_o)[0]
        return losses

    def loss_relations(self, outputs, targets, num_boxes, indices, log=True):
        assert 'rel_logits' in outputs
        src_logits = outputs['rel_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["rel_annotations"][J, 2] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_rel_classes, dtype=torch.int64,
                                    device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight_rel)
        # target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
        #                                     dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        # target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        # target_classes_onehot = target_classes_onehot[:, :, :-1]
        # loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=0.25, gamma=2) * \
        #           src_logits.shape[1]

        losses = {'loss_rel': loss_ce}
        if log:
            losses['rel_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, num_boxes, indices, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'relations': self.loss_relations
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, num_boxes, indices, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) + len(t["rel_annotations"]) * 2 for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, num_boxes, indices))
        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_rel_class(args):
    num_rel_classes = 51
    num_classes = 151
    position_embedding = PositionEmbeddingSine(128, normalize=True)
    backbone = Backbone('resnet50', False, False)
    backbone = Joiner(backbone, position_embedding)
    backbone.num_channels = 2048
    transformer = Transformer(d_model=256, dropout=0.1, nhead=8,
                              dim_feedforward=2048,
                              num_encoder_layers=6,
                              num_decoder_layers=6,
                              normalize_before=False)
    Match = RelTR(backbone, transformer, num_classes=151, num_entities=100, num_triplets=200)
    device = torch.device(args.device)

    # 加载了预训练权重
    ckpt = torch.load('./output/checkpoint0049.pth')['model']
    Match.load_state_dict(ckpt, strict=True)
    Match.eval()
    Match.to(device)
    # 加载预训练模型权重

    for name, para in Match.named_parameters():
        para.requires_grad_(False)
    RelModel = RelClass(256, num_rel_classes)
    matcher = build_matcher()
    losses = ['labels', "relations"]
    criterion = SetCriterion(num_classes, num_rel_classes, matcher=matcher, eos_coef=0.1, losses=losses)
    criterion.to(device)

    return Match, RelModel, criterion
