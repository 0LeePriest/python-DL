import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# ts1 = torch.randn(3,4)#默认放在cpu上
# print(ts1)
# ts2 = ts1.to("cuda:0")#转化为gpu上
# print(ts2)
#生成数据集
X1 = torch.rand(10000,1)
X2 = torch.rand(10000,1)
X3 = torch.rand(10000,1)
Y1 = ((X1+X2+X3)<1).float()
Y2 = ((1<(X1+X2+X3)) & ((X1+X2+X3)<2)).float()
Y3 = ((X1+X2+X3)>2).float()
Data = torch.cat([X1,X2,X3,Y1,Y2,Y3],axis = 1)
Data = Data.to("cuda:0")
#划分训练集和测试集
train_size = int(len(Data)*0.7)
test_size = len(Data) - train_size
Data = Data[torch.randperm(Data.size(0)),:]#打乱样本顺序
train_data = Data[:train_size,:]
test_data = Data[train_size:,:]
#搭建神经网络
class DNN(nn.Module):
    def __init__(self):
        super(DNN,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3,5),nn.ReLU(),
            nn.Linear(5,5),nn.ReLU(),
            nn.Linear(5,5),nn.ReLU(),
            nn.Linear(5,3)
        )
    def forward(self,x):
            y = self.net(x)
            return y

model = DNN().to("cuda:0")
# print(model.named_parameters())
#损失函数
loss_fun = nn.MSELoss()
#优化算法与学习率
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
#训练网络
epochs = 1000
losses = []
X = train_data[:,:3]  #前3列为输入特征
Y = train_data[:,-3:] #后3列为输出特征
for epoch in range(epochs):
    Pred = model(X)
    loss = loss_fun(Pred,Y)
    losses.append(loss.item())
    optimizer.zero_grad()  #清理上一轮的滞留参数
    loss.backward()
    optimizer.step()

Fig = plt.figure()
plt.plot(range(epochs),losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

#测试网络
X = test_data[:,:3]  #前3列为输入特征
Y = test_data[:,-3:] #后3列为输出特征
with torch.no_grad():
    Pred = model(X)
    Pred[:,torch.argmax(Pred,axis=1)]=1
    Pred[Pred!=1]=0
    correct = torch.sum((Pred==Y).all(1))
    total = Y.size(0)
    print(f"精度为{100*correct/total}%")


