import torch.nn as nn
import torch
import numpy as np


# input NCHW 1*4*140*63
# xyr,vxyr,xyR,vxyR
# output VxyR_(t+1) NCHW 1*1*5*7
#

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # input NCHW 1*4*140*63

        self.maxpool0 = nn.MaxPool2d(kernel_size=(2, 1), stride=(3, 1))

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(8, 16, 3, 2)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(16, 1, 3, 2)
        self.bn3 = nn.BatchNorm2d(1)
        self.relu3 = nn.ReLU(inplace=True)

        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.maxpool0(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = torch.flatten(x)
        x = self.softmax(x)
        x = torch.reshape(x, (1, 1, 5, 7))

        return x


class MonteCarloPolicyGradientLossFunc(nn.Module):
    def __init__(self, deta):
        super(MylossFunc, self).__init__()
        self.deta = deta

    def forward(self, out, label):
        out = torch.nn.functional.softmax(out, dim=1)
        m = torch.max(out, 1)[0]
        penalty = self.deta * torch.ones(m.size())
        loss = torch.where(m > 0.5, m, penalty)
        loss = torch.sum(loss)
        # loss = Variable(loss, requires_grad=True)
        return


if __name__ == "__main__":
    x = torch.tensor(np.random.random(size=(1, 4, 140, 63))).to(torch.float32).cuda()
    y = torch.tensor(np.zeros((1, 1, 5, 7))).to(torch.float32).cuda()  # , device='cuda'

    cnn = CNN().cuda()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    criterion = torch.nn.BCELoss()

    import time

    now = time.localtime()
    print(now)
    for i in range(100):
        predictions = cnn(x)

    print(predictions)
    now = time.localtime()
    print(now)

    loss = criterion(predictions, y)

    loss.backward()
    optimizer.step()
