## For Inference
:smile: It is super easy to configure the RelTR environment.
```
# create a conda environment 
conda create -n reltr python=3.7

conda activate reltr

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install matplotlib scipy pycocotools Cython tqdm

```

## Training/Evaluation on Visual Genome or Open Images V6
If you want to **train/evaluate** RelTR on Visual Genome, you need a little more preparation:

a) Scipy (we used 1.5.2) and pycocotools are required. 

b) Follow [README](https://github.com/yrcong/RelTR/blob/main/data/README.md) in the data directory to prepare the datasets.

c) Some widely-used evaluation code (**IoU**) need to be compiled... We will replace it with Pytorch code.
```
# compile the code computing box intersection
cd lib/fpn
sh make.sh
```

The directory structure looks like:
```
RelTR
| 
│
└───data
│   └───vg
│       │   rel.json
│       │   test.json
│       |   train.json
|       |   val.json
|       |   images
│   └───oi
│       │   rel.json
│       │   test.json
│       |   train.json
|       |   val.json
|       |   images
└───datasets    
... 
```

# 2. Usage

## Inference
a) Download our [RelTR model](https://drive.google.com/file/d/1id6oD_iwiNDD6HyCn2ORgRTIKkPD3tUD/view) pretrained on the Visual Genome dataset and put it under 
```
ckpt/checkpoint0149.pth
```
b) Infer the relationships in an image with the command:
```

python inference.py --img_path demo/customized.jpg --resume ckpt/checkpoint0149.pth

```

## Training
a) Train RelTR on Visual Genome on a single node with 8 GPUs (2 images per GPU):
```
export PYTHONUNBUFFERED=1

nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 --use_env main.py &


nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 --use_env rel_train.py &
 
```


## Evaluation
a) Evaluate the pretrained [RelTR](https://drive.google.com/file/d/1id6oD_iwiNDD6HyCn2ORgRTIKkPD3tUD/view) on Visual Genome with a single GPU (1 image per GPU):
```

nohup python main.py --eval --batch_size 1 --resume output/checkpoint0049.pth &


nohup python rel_train.py --eval --batch_size 8 --resume ./sample/train1/checkpoint0029.pth &


nohup python rel_test.py &

CUDA_VISIBLE_DEVICES=1 python rel_test.py
 
```

