import torch
import torch.nn as nn
import numpy as np
from DL_model.FCNet import FCNet

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())

x = np.arange(1, 36, 1)
x = torch.tensor(x)
x = torch.reshape(x, (1, 1, 5, 7))
print(x)
x = torch.flatten(x)
print(x)

torch.nn.MSELoss()

x = torch.empty(0,2)
print(x.shape)
print(x)

decision=FCNet()

torch.save({'state_dict': decision.state_dict()}, 'checkpoint_{:03d}.ckp'.format(1))
#torch.save({'state_dict': decision.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(1))