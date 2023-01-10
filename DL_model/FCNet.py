import torch.nn as nn
import torch


class FCNet(nn.Module):

    def __init__(self):
        super(FCNet, self).__init__()

        self.fc1 = nn.Linear(8, 8, bias=True)
        self.fc2 = nn.Linear(8, 8, bias=True)
        self.fc3 = nn.Linear(10, 8, bias=True)
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        x1, x2 = torch.split(x, 5, dim=0)
        x1 = self.sigmoid1(x1)
        x2 = self.sigmoid2(x2)
        x = torch.cat((x1, x2), dim=0)

        return x


if __name__ == "__main__":
    fcnet = FCNet()
