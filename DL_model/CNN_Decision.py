import torch
from DL_model import CNN


class CNNDecision:
    def __init__(self,
                 checkpointfile=None
                 ):
        self._net = CNN.CNN()
        self._crit = CNN.MonteCarloPolicyGradientLossFunc()
        self._optim = torch.optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

        if checkpointfile is not None:
            self.restore_checkpoint(checkpointfile)

    def restore_checkpoint(self, checkpointfile):
        ckp = torch.load(checkpointfile, 'cuda')
        self._net.load_state_dict(ckp['state_dict'])
