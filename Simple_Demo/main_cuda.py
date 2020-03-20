import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLoss(nn.Module):
    def __init(self):
        super(SimpleLoss, self).__init__()
    
    def forward(self, y_:torch.tensor, y:torch.tensor):
        a = torch.abs(y_ - y)
        l = torch.sum(a / 1000)
        return l

class Net(nn.Module):
    # 初始化
    def __init__(self):
        # 官方定义继承
        super(Net, self).__init__()
        # 定义全连接神经层
        self.fc1 = nn.Linear(3, 6)
        self.fc2 = nn.Linear(6, 1)
        # 定义l损失率和优化器
        self.loss = nn.L1Loss()
        #self.loss = SimpleLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
    
    # 定义神经网络
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    # 获取随机输入输出数据
def getRandomData(dataNum, device):
    inData = torch.rand(dataNum, 3, device=device) * 40 + 60
    outData = inData.sum(1)
    return (inData, outData)

    # 训练
def train(net:Net, trainNum:int, data:torch.tensor):
    (inData, outData) = data
    for i in range(trainNum):
        net.optimizer.zero_grad()
        output = net(inData)
        l = net.loss(output, outData)
        l.backward()
        net.optimizer.step()

        if (i % 100 == 0):
            print("Epoch: {}, loss = {}".format(i, l.item()))

if __name__ == "__main__":
    dev = torch.device("cuda")
    net = Net().cuda()
    train(net, 10000, getRandomData(1000, dev))

    # 定义一个向量
    a = torch.tensor([[1,2,3]], dtype=torch.float, device=dev)
    # 预测
    print(net(a))
