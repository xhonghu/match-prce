# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.

"""
dataset (COCO-like) which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import json
import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask()

        #TODO load relationship
        with open('/'.join(ann_file.split('/')[:-1])+'/rel.json', 'r') as f:
            all_rels = json.load(f)
        if 'train' in ann_file:
            self.rel_annotations = all_rels['train']
        elif 'val' in ann_file:
            self.rel_annotations = all_rels['val']
        else:
            self.rel_annotations = all_rels['test']

        self.rel_categories = all_rels['rel_categories']

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        rel_target = self.rel_annotations[str(image_id)]

        target = {'image_id': image_id, 'annotations': target, 'rel_annotations': rel_target}

        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


class ConvertCocoPolysToMask(object):
    '''
    主要工作是过滤掉标注为一组对象的数据，以及筛选掉bbox坐标不合法的那批数据。
    '''
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        # 将 "xywh" 转换为 "x1y1x2y2" 的形式
        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        #keep代表有效的bbox
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]
        rel_annotations = target['rel_annotations']

        #更新annotation里各个字段的值，同时新增 'orig_size ' 和 'size' 两个 key，最后返回处理后的图像和标签
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        target['rel_annotations'] = torch.tensor(rel_annotations)

        return image, target


def make_coco_transforms(image_set):
    '''
    数据增强部分
    '''

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    # TODO: cropping causes that some boxes are dropped then no tensor in the relation part!
                    # TODO:What should we do?
                    #T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    ann_path = args.ann_path
    img_folder = args.img_folder
    if image_set == 'train':
        ann_file = ann_path + 'train.json'
    elif image_set == 'val':
        if args.eval:
            ann_file = ann_path + 'test.json'
        else:
            ann_file = ann_path + 'val.json'

    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set))
    return dataset
