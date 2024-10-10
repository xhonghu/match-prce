import json
import matplotlib.pyplot as plt
from collections import Counter

REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
               'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
               'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
               'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
               'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
               'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

with open('rel.json') as file:
    rel = json.load(file)

relation = []
for k in rel['train']:
    for i in rel['train'][k]:
        relation.append(i[2])
counter = Counter(relation)


def draw_from_dict(dicdata, RANGE):
    # dicdata：字典的数据。
    # RANGE：截取显示的字典的长度。
    plt.figure(figsize=(15, 6))
    by_value = sorted(dicdata.items(), key=lambda item: item[1], reverse=True)
    x = []
    y = []
    for d in by_value:
        x.append(REL_CLASSES[d[0]])
        y.append(d[1])
        the_class = REL_CLASSES[d[0]]
        print(f"{the_class}:          {d[1]}")
    plt.barh(x[0:RANGE], y[0:RANGE])
    # 在每个柱子上添加具体的值
    for i in range(RANGE):
        plt.text(y[i], i, str(y[i]), ha='left', va='center')
    plt.title("Relationship Labels Distribution Chart of VG (Top20)")
    plt.show()
    return


draw_from_dict(dict(counter), 20)
