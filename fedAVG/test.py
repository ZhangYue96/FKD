import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset, DataLoader
from net import *
from dataUtils import *


net_glob = CNNMnist()
params = net_glob.state_dict()
b = torch.range(1,10,2)
# print(b)
# print(params)

print(torch.__version__)
print(torch.version.cuda)