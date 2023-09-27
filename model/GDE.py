import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model import Gradient
from model.Blocks import conv1x1, conv3x3, ResBlock

class SFE(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale = 1):
        super(SFE, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv_head = conv3x3(3, n_feats)
        
        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats, 
                res_scale=res_scale))
            
        self.conv_tail = conv3x3(n_feats, n_feats)
        
    def forward(self, x):
        x  = F.relu(self.conv_head(x))
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        x = x + x1
        return x

class GDE(nn.Module):
    def __init__(self, num_grad_blocks, n_feats):
        super(GDE, self).__init__()
        self.gradient = Gradient.gradient()
        self.num_res_blocks_g = num_grad_blocks

        self.SFE_GRAD    = SFE(self.num_res_blocks_g[0], n_feats)        
       
        self.conv12_grad = conv3x3(2 * n_feats, n_feats)
        self.grad_12     = nn.ModuleList()
        for i in range(self.num_res_blocks_g[1]):
            self.grad_12.append(ResBlock(in_channels  = n_feats, out_channels = n_feats))
                                
        self.conv23_grad = conv3x3(2 * n_feats, n_feats)
        self.grad_23     = nn.ModuleList()
        for i in range(self.num_res_blocks_g[2]):
            self.grad_23.append(ResBlock(in_channels  = n_feats, out_channels = n_feats))
        
        self.conv33_grad = conv3x3(2 * n_feats, n_feats)
        self.grad_33     = nn.ModuleList()
        for i in range(self.num_res_blocks_g[3]):
            self.grad_33.append(ResBlock(in_channels  = n_feats, out_channels = n_feats))

        self.fuse        = conv3x3(2 * n_feats, n_feats)
                                
        self.fuse_tail1 = conv3x3(n_feats, n_feats//2)
        self.fuse_tail2 = conv1x1(n_feats//2, 3)

    def forward(self, x, x_tt = None, T3 = None, T2 = None, T1 = None):
        g      = self.gradient((x + 1) * 127.5)
        ### shallow feature extraction
        x_grad = self.SFE_GRAD(g)
                
        # fuse level 1
        x_grad1 = torch.cat([x_grad, T1], dim = 1)
        x_grad1 = F.relu(self.conv12_grad(x_grad1))
        
        x_grad1_res = x_grad1
        for i in range(self.num_res_blocks_g[1]):
            x_grad1_res = self.grad_12[i](x_grad1_res)
        x_grad1 = x_grad1 + x_grad1_res

        # fuse level 2
        x_grad1 = F.interpolate(x_grad1, scale_factor = 2, mode='bicubic')
        x_grad2 = torch.cat([x_grad1, T2], dim = 1)
        x_grad2 = F.relu(self.conv23_grad(x_grad2))

        x_grad2_res = x_grad2
        for i in range(self.num_res_blocks_g[2]):
            x_grad2_res = self.grad_23[i](x_grad2_res)
        x_grad2 = x_grad2 + x_grad2_res

        # fuse level 3
        x_grad2 = F.interpolate(x_grad2, scale_factor = 2, mode='bicubic')
        x_grad3 = torch.cat([x_grad2, T3], dim = 1)
        x_grad3 = F.relu(self.conv33_grad(x_grad3))
        
        x_grad3_res = x_grad3
        for i in range(self.num_res_blocks_g[3]):
            x_grad3_res = self.grad_33[i](x_grad3_res)
        x_grad3 = x_grad3 + x_grad3_res

        x_cat = torch.cat([x_tt, x_grad3], dim = 1)
        x_cat = F.relu(self.fuse(x_cat))
        
        x_cat = self.fuse_tail1(x_cat)
        x_cat = self.fuse_tail2(x_cat)
        
        return torch.clamp(x_cat, -1, 1)  