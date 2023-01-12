import torch
import torch.nn as nn

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())

m = nn.Softmax(dim=(2,3))
# you softmax over the 2nd dimension
input = torch.randn((1,1,2,3))
print(m(input))