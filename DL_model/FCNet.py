import torch.nn as nn
import torch
import numpy as np


# input 8
# xr,yr,vxr,vyr,xR,yR,vxR,vyR
# output 10
#

class FCNet(nn.Module):

    def __init__(self):
        super(FCNet, self).__init__()

        self.fc1 = nn.Linear(8, 8, bias=True)
        self.fc2 = nn.Linear(8, 8, bias=True)
        self.fc3 = nn.Linear(8, 10, bias=True)
        self.softmax1 = nn.Softmax(dim=0)
        self.softmax2 = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        x1, x2 = torch.split(x, 5, dim=0)
        x1 = self.softmax1(x1)
        x2 = self.softmax2(x2)
        x = torch.cat((x1, x2), dim=0)

        return x


if __name__ == "__main__":
    x = torch.tensor(np.ones(8)).to(torch.float32)
    y = torch.tensor(np.zeros(10)).to(torch.float32)  # , device='cuda'

    fcnet = FCNet()
    optimizer = torch.optim.Adam(fcnet.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    criterion = torch.nn.BCELoss()

    predictions = fcnet(x)

    print(predictions)

    loss = criterion(predictions, y)

    loss.backward()
    optimizer.step()
