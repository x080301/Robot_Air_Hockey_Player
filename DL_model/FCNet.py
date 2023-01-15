import torch.nn as nn
import torch
import numpy as np
# from defensive_control_strategy.defensive_control_strategy import Parameters
from physical_model.Parameters import Parameters


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

        self.loss_function = MonteCarloPolicyGradientLossFunc()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        x = x / 100
        x = self.act(x)
        # print(x)

        x = torch.mul(x, Parameters.Striker.Max_velocity)

        # print(x)
        """
        x1, x2 = torch.split(x, 5, dim=0)
        x1 = self.softmax1(x1)
        x2 = self.softmax2(x2)
        x = torch.cat((x1, x2), dim=0)
        """
        # x=torch.matmul(x1, x2.T, out=None)

        return x


class MonteCarloPolicyGradientLossFunc(nn.Module):
    # TODO: Do this part after modifying the physical model. Now the result is only a punishment.
    def __init__(self):
        super(MonteCarloPolicyGradientLossFunc, self).__init__()

    def forward(self, y, action, reward):
        return nn.functional.mse_loss(action, y) * reward
        # return 10.0 + Parameters.Puck._radius + Parameters.Striker._radius - nn.functional.mse_loss(action, y) * reward


if __name__ == "__main__":
    x = torch.tensor(np.ones(8)).to(torch.float32).cuda()
    x = torch.reshape(x, (1, 8))
    y = torch.tensor(np.zeros(2)).to(torch.float32).cuda()  # , device='cuda'
    y = torch.reshape(y, (1, 2))

    fcnet = FCNet().cuda()
    optimizer = torch.optim.Adam(fcnet.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    criterion = MonteCarloPolicyGradientLossFunc()

    predictions = fcnet(x)

    print(predictions.shape)

    '''loss = criterion(predictions, y)

    loss.backward()
    print(loss)
    print(loss.backward)
    optimizer.step()

    a = predictions[1]
    print(a)'''
