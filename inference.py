import argparse
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from models.backbone import Backbone, Joiner
from models.position_encoding import PositionEmbeddingSine
from models.transformer import Transformer
from models.reltr import RelTR

# VG classes
CLASSES = ['N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
           'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
           'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
           'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
           'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
           'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
           'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
           'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
           'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
           'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
           'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
           'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
           'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
           'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # image path
    parser.add_argument('--img_path', type=str, default='demo/vg1.jpg',
                        help="Path of the test image")
    parser.add_argument('--resume', default='ckpt/checkpoint0149.pth', help='resume from checkpoint')

    return parser


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def main(args):
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    position_embedding = PositionEmbeddingSine(128, normalize=True)
    backbone = Backbone('resnet50', False, False)
    backbone = Joiner(backbone, position_embedding)
    backbone.num_channels = 2048
    transformer = Transformer(d_model=256, dropout=0.1, nhead=8,
                              dim_feedforward=2048,
                              num_encoder_layers=6,
                              num_decoder_layers=6,
                              normalize_before=False)
    model = RelTR(backbone, transformer, num_classes=151, num_entities=100, num_triplets=200)

    ## 加载预训练模型的权重
    ckpt = torch.load(args.resume)['model']
    model2_dict = model.state_dict()
    state_dict = {k: v for k, v in ckpt.items() if k in model2_dict.keys()}
    model2_dict.update(state_dict)
    model.load_state_dict(model2_dict)
    model.eval()
    im = Image.open(args.img_path)
    print(im.size)
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)
    # propagate through the model
    outputs, hs_t, so_masks, memory= model(img)

    # keep only predictions with 0.+ confidence
    probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
    print(outputs['sub_logits'].shape)
    print(probas_sub.shape)
    probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]
    print(outputs['obj_logits'].shape)
    print(probas_obj.shape)
    # 当两个阈值都大于0.3时才有效
    keep = torch.logical_and(probas_sub.max(-1).values > 0.3, probas_obj.max(-1).values > 0.3)
    print(keep)
    # convert boxes from [0; 1] to image scales
    sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][0, keep], im.size)
    obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][0, keep], im.size)

    ##决定识别出几个关系出来
    topk = 10
    ##keep_queries决定了200组查询中满足条件的所有查询的下标
    keep_queries = torch.nonzero(keep, as_tuple=True)[0]
    ##torch.agrsort是从小到大排序的，这里是主谓宾三者的置信度相乘排序，加负号是为了从大到小排序
    indices = torch.argsort(-probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[0])[:topk]
    keep_queries = keep_queries[indices]

    with torch.no_grad():
        fig, axs = plt.subplots(ncols=len(indices), nrows=1, figsize=(22, 7))
        for idx, ax_i, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in \
                zip(keep_queries, axs.T, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices]):
            ax = ax_i
            ax.imshow(im)
            ax.add_patch(plt.Rectangle((sxmin, symin), sxmax - sxmin, symax - symin,
                                       fill=False, color='blue', linewidth=2.5))
            ax.add_patch(plt.Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin,
                                       fill=False, color='orange', linewidth=2.5))
            ax.axis('off')
            ##argmax返回最大值的索引。
            ax.set_title(CLASSES[probas_sub[idx].argmax()] + '       ' + CLASSES[probas_obj[idx].argmax()], fontsize=10)
        fig.tight_layout()
        plt.show()
        plt.savefig('demo/output.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RelTR inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
