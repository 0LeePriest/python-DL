import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
# from datasets import load_from_disk
from transformers import BertTokenizer, BertModel, AdamW
from transformers.optimization import get_scheduler
from tqdm import tqdm
# x = torch.tensor([1, 2, 3, 4])
# y = torch.unsqueeze(x, 0)#在第0维扩展，第0维大小为1
# print(x.shape)
# print(y.shape)
# print("*********")
# y = torch.unsqueeze(x, 1)#在第1维扩展，第1维大小为1
# print(x.shape)
# print(y.shape)
df = pd.read_csv("C:\\Users\\23993\\Desktop\\data.csv", encoding="ANSI")
datalist = df.values.tolist()
random.shuffle(datalist)
# train_data = datalist[:int(len(datalist)*0.8)]# 训练数据列表，每个元素包含(text, label)


# val_data = datalist[int(len(datalist)*0.8):int(len(datalist)*0.9)]  # 验证数据列表
test_data = datalist[int(len(datalist)*0.9):]  # 测试数据列表
test_text = []
text_label =[]
for i in test_data:
    test_text.append(i[1])
    text_label.append(i[2])
test_data = (test_text,text_label)
print(test_data)
# print(train_data[0][1])
# print(val_data[0][1])
# print(test_data)
# train_texts = [
#     "这部电影太棒了！",
#     "情节拖沓，表演生硬。",
#     "视觉效果震撼。",
#     "失望透顶。",
#     "一般般，没什么特别的。",
#     "推荐给所有科幻迷！",
#     "剧本很有创意。",
#     "演员演技炸裂。",
#     "不太理解为什么这么高评分。",
#     "意料之外的好看！"
# ]

# train_labels = [1, 0, 1, 0, 0, 1, 1, 1, 0, 1]  # 1代表正面评价，0代表负面评价
#
# train_data = (train_texts, train_labels)
# print(train_data[1])

