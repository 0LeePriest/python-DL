import torch
import pandas as pd
#创建一维series对象
# k =['a','b','c']
# v1=  [1,2,3]
# sr = pd.Series(v1,index=k)
# print(sr)
# print(sr.values)
#创建二维，即dataframe对象
v2 = [[53,'女'],[64,'男'],[70,'女']]
i = ['1','2','3']
c = ['age','gender']
df = pd.DataFrame(v2,index=i,columns=c)
print(df)
v1=  [11,22,33]
sr = pd.Series(v1,index=i)
# 新加入一列
df['牌照'] = sr
df['new'] = [6,6,6]
print(df)
#新加入一行
# df.loc['4'] = [56,'男','22',5]
# print(df)
# dfva = df.values[:,0].astype(int)
# print(dfva)
# print(df.loc['1','age'])
# print(df.loc[['2','3'],['age','gender']])
# print(df.T)
# df = pd.read_csv('D:\waimai_train.csv',index_col=0,encoding='ANSI')
# print(df.head())


# 创建数据
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 22],
    'City': ['New York', 'London', 'Paris']
}

df = pd.DataFrame(data)
print(df)
# 导出到 CSV 文件
df.to_csv('output.csv', index=False)

