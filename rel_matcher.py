# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.
"""
Modules to compute the matching cost between the predicted triplet and ground truth triplet.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network"""

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, cost_rel: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_rel = cost_rel
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries_rel = outputs["rel_logits"].shape[:2]
        alpha = 0.25
        gamma = 2.0
        # Concat the subject/object/predicate labels and subject/object boxes
        sub_tgt_bbox = torch.cat([v['boxes'][v['rel_annotations'][:, 0]] for v in targets])
        sub_tgt_ids = torch.cat([v['labels'][v['rel_annotations'][:, 0]] for v in targets])
        obj_tgt_bbox = torch.cat([v['boxes'][v['rel_annotations'][:, 1]] for v in targets])
        obj_tgt_ids = torch.cat([v['labels'][v['rel_annotations'][:, 1]] for v in targets])
        rel_tgt_ids = torch.cat([v["rel_annotations"][:, 2] for v in targets])

        sub_prob = outputs["sub_logits"].flatten(0, 1).sigmoid()
        sub_bbox = outputs["sub_boxes"].flatten(0, 1)
        obj_prob = outputs["obj_logits"].flatten(0, 1).sigmoid()
        obj_bbox = outputs["obj_boxes"].flatten(0, 1)
        rel_prob = outputs["rel_logits"].flatten(0, 1).sigmoid()

        # Compute the subject matching cost based on class and box.
        neg_cost_class_sub = (1 - alpha) * (sub_prob ** gamma) * (-(1 - sub_prob + 1e-8).log())
        pos_cost_class_sub = alpha * ((1 - sub_prob) ** gamma) * (-(sub_prob + 1e-8).log())
        cost_sub_class = pos_cost_class_sub[:, sub_tgt_ids] - neg_cost_class_sub[:, sub_tgt_ids]
        cost_sub_bbox = torch.cdist(sub_bbox, sub_tgt_bbox, p=1)
        cost_sub_giou = -generalized_box_iou(box_cxcywh_to_xyxy(sub_bbox), box_cxcywh_to_xyxy(sub_tgt_bbox))

        # Compute the object matching cost based on class and box.
        neg_cost_class_obj = (1 - alpha) * (obj_prob ** gamma) * (-(1 - obj_prob + 1e-8).log())
        pos_cost_class_obj = alpha * ((1 - obj_prob) ** gamma) * (-(obj_prob + 1e-8).log())
        cost_obj_class = pos_cost_class_obj[:, obj_tgt_ids] - neg_cost_class_obj[:, obj_tgt_ids]
        cost_obj_bbox = torch.cdist(obj_bbox, obj_tgt_bbox, p=1)
        cost_obj_giou = -generalized_box_iou(box_cxcywh_to_xyxy(obj_bbox), box_cxcywh_to_xyxy(obj_tgt_bbox))

        # Compute the object matching cost only based on class.
        neg_cost_class_rel = (1 - alpha) * (rel_prob ** gamma) * (-(1 - rel_prob + 1e-8).log())
        pos_cost_class_rel = alpha * ((1 - rel_prob) ** gamma) * (-(rel_prob + 1e-8).log())
        cost_rel_class = pos_cost_class_rel[:, rel_tgt_ids] - neg_cost_class_rel[:, rel_tgt_ids]

        # Final triplet cost matrix
        # C_rel = self.cost_bbox * cost_sub_bbox + self.cost_bbox * cost_obj_bbox + \
        #         self.cost_class * cost_sub_class + self.cost_class * cost_obj_class + self.cost_rel * cost_rel_class + \
        #         self.cost_giou * cost_sub_giou + self.cost_giou * cost_obj_giou
        C_rel = self.cost_class * cost_sub_class + self.cost_class * cost_obj_class + self.cost_rel * cost_rel_class
        C_rel = C_rel.view(bs, num_queries_rel, -1).cpu()

        sizes1 = [len(v["rel_annotations"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C_rel.split(sizes1, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher():
    return HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2, cost_rel=1)
