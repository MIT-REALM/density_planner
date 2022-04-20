import torch
from systems.sytem_CAR import Car
from systems.ControlAffineSystem import ControlAffineSystem
import sys
from utils import approximate_derivative


N = [7, 5, 3, 3]
dim_x = len(N)

x = torch.arange(N[0]).unsqueeze(-1)
for i in range(1, dim_x):
    leng_x = x.shape[0]
    for j in range(N[i]):
        if j == 0:
            y = j * torch.ones(leng_x, 1)
        else:
            y = torch.cat((y, j * torch.ones(leng_x, 1)), 0)
    x = torch.cat((x.repeat(N[i], 1), y), 1)


print("end")