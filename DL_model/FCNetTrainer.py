from . import FCNet
import torch

class FCNetTrainer:

    def __init__(self):
        self.net=FCNet.FCNet().cuda()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
        self.criterion = torch.nn.BCELoss()#TODO



if __name__ == "__main__":
    pass

