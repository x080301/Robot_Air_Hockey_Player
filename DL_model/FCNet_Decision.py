from . import FCNet
import torch


class Decision:

    def __init__(self,
                 checkpointfile=None
                 ):
        self.net = FCNet.FCNet().cuda()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
        self.criterion = torch.nn.BCELoss()  # TODO

        if checkpointfile is not None:
            self.restore_checkpoint(checkpointfile)

    def restore_checkpoint(self, checkpointfile):
        ckp = torch.load(checkpointfile, 'cuda')
        self._net.load_state_dict(ckp['state_dict'])


if __name__ == "__main__":
    pass
