import torch.nn as nn
import torch
import numpy as np


# input 8
# xr,yr,vxr,vyr,xR,yR,vxR,vyR
# output Vx_t+1,Vy_t+1
#

class FCNet(nn.Module):

    def __init__(self):
        super(FCNet, self).__init__()

        self.fc1 = nn.Linear(8, 8, bias=True)
        self.fc2 = nn.Linear(8, 8, bias=True)
        self.fc3 = nn.Linear(8, 2, bias=True)

        self.act = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        x = self.act(x)
        """
        x1, x2 = torch.split(x, 5, dim=0)
        x1 = self.softmax1(x1)
        x2 = self.softmax2(x2)
        x = torch.cat((x1, x2), dim=0)
        """
        # x=torch.matmul(x1, x2.T, out=None)

        return x


class MylossFunc(nn.Module):
    def __init__(self, deta):
        super(MylossFunc, self).__init__()
        self.deta = deta

    def forward(self, out, label):
        out = torch.nn.functional.softmax(out, dim=1)
        m = torch.max(out, 1)[0]
        penalty = self.deta * torch.ones(m.size())
        loss = torch.where(m > 0.5, m, penalty)
        loss = torch.sum(loss)
        loss = Variable(loss, requires_grad=True)
        return


if __name__ == "__main__":
    x = torch.tensor(np.ones(8)).to(torch.float32).cuda()
    y = torch.tensor(np.zeros(2)).to(torch.float32).cuda()  # , device='cuda'

    fcnet = FCNet().cuda()
    optimizer = torch.optim.Adam(fcnet.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    criterion = torch.nn.MSELoss()

    predictions = fcnet(x)

    print(predictions)

    loss = criterion(predictions, y)

    loss.backward()
    optimizer.step()

    a=predictions[1]
    print(a)
