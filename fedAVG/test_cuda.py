import torch

if torch.cuda.is_available():
    device = torch.device("cuda")          # 如果GPU可用，则使用CUDA
else:
    device = torch.device("cpu")           # 如果GPU不可用，则使用CPU

print(device)