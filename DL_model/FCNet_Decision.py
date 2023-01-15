import torch
from DL_model import FCNet


class Decision:

    def __init__(self,
                 checkpointfile=None
                 ):
        self.net = FCNet.FCNet().cuda()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

        if checkpointfile is not None:
            self.restore_checkpoint(checkpointfile)




if __name__ == "__main__":
    pass
