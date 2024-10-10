## For Inference
:smile: It is super easy to configure the environment.
```
# create a conda environment 
conda create -n match python=3.7

conda activate match

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install matplotlib scipy pycocotools Cython tqdm

```

## Training/Evaluation on Visual Genome or Open Images V6
If you want to **train/evaluate** match on Visual Genome, you need a little more preparation:

a) Scipy (we used 1.5.2) and pycocotools are required. 

b) Follow [README](https://github.com/yrcong/match/blob/main/data/README.md) in the data directory to prepare the datasets.

c) Some widely-used evaluation code (**IoU**) need to be compiled... We will replace it with Pytorch code.
```
# compile the code computing box intersection
cd lib/fpn
sh make.sh
```

The directory structure looks like:
```
match-prce
| 
└───data
│   └───vg
|       ├── images
|       ├── vg_init
|       │   ├── rel.json
|       │   ├── test.json
|       │   ├── train.json
|       │   └── val.json
|       ├── vg_sample_1
|       │   ├── annotation_modify.py
|       │   ├── rel.json
|       │   ├── rel_lable_distribution.py
|       │   ├── test.json
|       │   ├── train.json
|       │   └── val.json
|       ├── vg_sample_2
|       ├── vg_sample_3
|       └── vg_sample_4
|───datasets
|───lib
|───models
|───util
├── engine.py
├── README.md
├── rel_class.py
├── rel_engine.py
├── rel_matcher.py
├── rel_test.py
├── rel_train.py     
... 
```

# 2. Usage

## Training
a) Train match on Visual Genome on a single node with 2 GPUs (2 images per GPU):
```
# Train the match model
nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 --use_env main.py &

# Train predicate classifier
nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 --use_env rel_train.py --ann_path path_to_sample &
 
```


## Evaluation
a) Evaluate the pretrained on Visual Genome with a single GPU (1 image per GPU):
```
# The first step of the test is to match the object detection of the subject and object

python main.py --eval --batch_size 1 --resume path_to_Match.pth

# Test the performance of each predicate classifier

python rel_train.py --eval --batch_size 1 --resume path_to_classifier.pth

#Test the performance of your integration model


python rel_test.py
```

