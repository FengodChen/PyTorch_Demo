import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class WinnerLoss(nn.Module):
    def __init(self):
        super(WinnerLoss, self).__init__()
    
    def forward(self, y:torch.tensor):
        return torch.sum(y)

class Net(nn.Module):
    def __init__(self, size:(int, int)):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(size[0]*size[1], 80)
        self.fc2 = nn.Linear(80, size[0]*size[1])

        self.loss = WinnerLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class AI:
    def __init__(self, player):
        self.player = player
        self.size = player.size
        self.net = Net(self.size)

    def findLocation(self):
        inData = self.player.getPlayerView()
        inData = inData.flatten()
        inData = torch.tensor(inData, dtype=torch.float32)
        output = self.net(inData)
        output = output.reshape(self.size)
        zeroArea = self.player.getEmptyArea()
        zeroArea = torch.tensor(zeroArea)
        output[torch.where(zeroArea == 0)] = float('-inf')
        m = int(output.argmax())
        ptr = (m // self.size[1], m % self.size[1])
        return ptr
    
    def train(self, board):
        self.net.optimizer.zero_grad()
        l = self.net.loss(board)
        l.backward()
        self.net.optimizer.step()
