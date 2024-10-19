# import torch
# # 假设是时间步T1的输出
# T1 = torch.tensor([[1, 2, 3],
#         		[4, 5, 6],
#         		[7, 8, 9]])
# # 假设是时间步T2的输出
# T2 = torch.tensor([[10, 20, 30],
#         		[40, 50, 60],
#         		[70, 80, 90]])
# print(torch.stack((T1,T2),dim=0).shape)
# print(torch.stack((T1,T2),dim=0))
# print(torch.stack((T1,T2),dim=1).shape)
# print(torch.stack((T1,T2),dim=1))
# print(torch.stack((T1,T2),dim=2).shape)
# print(torch.stack((T1,T2),dim=2))
# a = [1, 2, 3]
# b = [4, 5, 6]
# c = [7, 8, 9, 10, 11, 12]
# print(*list(zip(a, c)))
# zip_object = zip(a, b)
# print(*a)
# print(zip_object)
# print(*zip_object)
list1 = ["这", "是", "一个", "测试"]
for i,j in enumerate(list1):
    print(i,j)
print(list(enumerate(list1)))







