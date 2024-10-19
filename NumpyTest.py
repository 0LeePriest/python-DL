import numpy as np
#创建数组 np.array(传个list)，ones，arange,zeros,random.ranom(0-1均匀分布的数组)，random.randint(整数随机数组)
#random.normal(正太分布)
#转变类型 astype
#形状相关 shape,reshape
#数组的翻转np.flipud,fliplr(向量只能上下翻转)
#向量的拼接np.concatenate
#向量的分割 np.split
#向量进行乘积 np.dot(是线性代数的乘积)
#常用函数 np.abs,三角函数,对数函数，指数函数,max,sum,mean,std
#布尔类型的数组
# arr1 = np.arange(4)
# arr2 = np.flipud(arr1)
# print(arr1>arr2)
# [False False  True  True]
#统计布尔数组里True的数量 np.sum(bool数组)
#np.any()只要bool有True就返回True
#bool数组可进行掩码
import torch
print(torch.cuda.is_available())