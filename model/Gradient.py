import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class gradient(nn.Module):
    def __init__(self):
        super(gradient, self).__init__()
        # Sober Filter
        kernel_v = [[1,   2,  1],
                    [0,   0,  0],
                    [-1, -2, -1]]
        kernel_h = [[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]]
        
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0).cuda()
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0).cuda()
        
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding = 1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding = 1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding = 1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding = 1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding = 1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding = 1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)
        
        x0 = (x0 - x0.min()) / (x0.max() - x0.min())
        x1 = (x1 - x1.min()) / (x1.max() - x1.min())
        x2 = (x2 - x2.min()) / (x2.max() - x2.min())
        
        x = torch.cat([x0, x1, x2], dim=1)
        return x