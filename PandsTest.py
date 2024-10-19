import pandas as pd
import numpy as np
# df = pd.DataFrame({"id":[1001,1002,1003,1004,1005,1006],
#  "date":pd.date_range('20130102', periods=6),
#   "city":['Beijing ', 'SH', ' guangzhou ', 'Shenzhen', 'shanghai', 'BEIJING '],
#  "age":[23,44,54,32,34,32],
#  "category":['100-A','100-B','110-A','110-C','210-A','130-F'],
#   "price":[1200,np.nan,2133,5433,np.nan,4432]},
#   columns =['id','date','city','category','age','price'])
# df.replace(1002,1008,inplace=True)
# print(df)
# print(df['city'][1])
# # print(df.values[:,0])
# df1 = pd.read_csv('D:\waimai_train.csv',encoding='ANSI',sep=',')
# # # print(df1.head())
# df2 = pd.read_csv('D:\shopping_train.csv',encoding='ANSI',sep=',',usecols=["review", "label"])
# # # print(df2.head())
# df3 = pd.read_csv('D:\movie_train.csv',encoding='ANSI',sep=',',usecols=["comment", "rating"])
# df3['rating'].replace([1,5],[0,1],inplace=True)
# df3.rename(columns={'rating': 'label'},inplace=True)
# df3.rename(columns={'comment': 'review'},inplace=True)
# df4 = pd.concat([df1,df2,df3])
# df4.to_csv('output.csv')
df1 = pd.read_csv('output.csv',encoding='ANSI',index_col=0)
# df1.index.name="编号"

print(df1.values)
# print(df1.tail())
# df2 = pd.DataFrame(df1,columns=["review","label"])




