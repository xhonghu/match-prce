import json
import random
from tqdm import tqdm


def random_sample(original_list, random_seed):
    random.seed(random_seed)
    new_list = []
    # 使用循环从原始列表中随机选择元素，并添加到新列表中
    for _ in range(len(original_list)):
        random_element = random.choice(original_list)
        new_list.append(random_element)
    return new_list


REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
               'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
               'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
               'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
               'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
               'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']
a = REL_CLASSES.index('on')
b = REL_CLASSES.index('has')
c = REL_CLASSES.index('wearing')
d = REL_CLASSES.index('of')
e = REL_CLASSES.index('in')
f = REL_CLASSES.index('near')
g = REL_CLASSES.index('behind')
h = REL_CLASSES.index('with')
low_frequency = tuple({i for i in range(51)} - {a, b, c, d, e, f, g, h})

with open('rel.json') as file:
    rel = json.load(file)
with open('train.json', 'r') as file:
    init_train = json.load(file)

id = []
id_high_frequency = []
for k, v in rel['train'].items():
    for i in v:
        if i[2] in low_frequency:
            id.append(k)
            break
    else:
        id_high_frequency.append(k)

# 设置随机数种子
seed = 42
random.seed(seed)
id_choice_high = random_sample(id_high_frequency, seed)
print("尾部有以下数量",len(id))
print("头部有以下数量",len(id_high_frequency))
all_id = id + id_choice_high
del_list = list(set(id_high_frequency) - set(id_choice_high))
print("有以下数量的被去除了",len(del_list))
all_id = id + list(set(id_choice_high))
random.shuffle(all_id)


# 去除一些没被抽样到的头部关系
for index in del_list:
    del rel['train'][str(index)]
# 修改train标注
train = {'images': [], 'annotations': [], 'categories': init_train['categories'][:]}
for index in tqdm(all_id):
    for image in init_train['images'][:]:
        if int(image['id']) == int(index):
            train['images'].append(image)
    for annotation in init_train['annotations'][:]:
        if int(annotation['image_id']) == int(index):
            train['annotations'].append(annotation)

with open('train.json', 'w') as file:
    json.dump(train, file)
with open('rel.json', 'w') as file:
    json.dump(rel, file)
