import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model import Gradient
from model.Blocks import conv1x1, conv3x3, ResBlock


# thanks to
# https://github.com/researchmm/TTSR

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

class CSFI2(nn.Module):
    def __init__(self, n_feats):
        super(CSFI2, self).__init__()
        self.conv12 = conv1x1(n_feats, n_feats)
        self.conv21 = conv3x3(n_feats, n_feats, 2)

        self.conv_merge1 = conv3x3(n_feats*2, n_feats)
        self.conv_merge2 = conv3x3(n_feats*2, n_feats)

    def forward(self, x1, x2):
        x12 = F.interpolate(x1, scale_factor=2, mode='bicubic')
        x12 = F.relu(self.conv12(x12))
        x21 = F.relu(self.conv21(x2))

        x1 = F.relu(self.conv_merge1( torch.cat((x1, x21), dim=1) ))
        x2 = F.relu(self.conv_merge2( torch.cat((x2, x12), dim=1) ))

        return x1, x2


class CSFI3(nn.Module):
    def __init__(self, n_feats):
        super(CSFI3, self).__init__()
        self.conv12 = conv1x1(n_feats, n_feats)
        self.conv13 = conv1x1(n_feats, n_feats)

        self.conv21 = conv3x3(n_feats, n_feats, 2)
        self.conv23 = conv1x1(n_feats, n_feats)

        self.conv31_1 = conv3x3(n_feats, n_feats, 2)
        self.conv31_2 = conv3x3(n_feats, n_feats, 2)
        self.conv32 = conv3x3(n_feats, n_feats, 2)

        self.conv_merge1 = conv3x3(n_feats*3, n_feats)
        self.conv_merge2 = conv3x3(n_feats*3, n_feats)
        self.conv_merge3 = conv3x3(n_feats*3, n_feats)

    def forward(self, x1, x2, x3):
        x12 = F.interpolate(x1, scale_factor=2, mode='bicubic')
        x12 = F.relu(self.conv12(x12))
        x13 = F.interpolate(x1, scale_factor=4, mode='bicubic')
        x13 = F.relu(self.conv13(x13))

        x21 = F.relu(self.conv21(x2))
        x23 = F.interpolate(x2, scale_factor=2, mode='bicubic')
        x23 = F.relu(self.conv23(x23))

        x31 = F.relu(self.conv31_1(x3))
        x31 = F.relu(self.conv31_2(x31))
        x32 = F.relu(self.conv32(x3))

        x1 = F.relu(self.conv_merge1( torch.cat((x1, x21, x31), dim=1) ))
        x2 = F.relu(self.conv_merge2( torch.cat((x2, x12, x32), dim=1) ))
        x3 = F.relu(self.conv_merge3( torch.cat((x3, x13, x23), dim=1) ))
        
        return x1, x2, x3

class MergeTail(nn.Module):
    def __init__(self, n_feats):
        super(MergeTail, self).__init__()
        self.conv13     = conv1x1(n_feats, n_feats)
        self.conv23     = conv1x1(n_feats, n_feats)
        self.conv_merge = conv3x3(n_feats*3, n_feats)
        
    def forward(self, x1, x2, x3):
        x13 = F.interpolate(x1, scale_factor=4, mode='bicubic')
        x13 = F.relu(self.conv13(x13))
        x23 = F.interpolate(x2, scale_factor=2, mode='bicubic')
        x23 = F.relu(self.conv23(x23))

        x = F.relu(self.conv_merge( torch.cat((x3, x13, x23), dim=1) ))
        
        return x

class CSFI(nn.Module):
    def __init__(self, num_res_blocks, n_feats, top_k):
        super(CSFI, self).__init__()
        self.num_res_blocks   = num_res_blocks
        self.n_feats          = n_feats
        self.top_k            = top_k

        self.SFE            = SFE(self.num_res_blocks[0], n_feats)
        ### stage11
        self.conv11_head = nn.ModuleList()
        for i in range(top_k):
            self.conv11_head.append(conv3x3(256 + n_feats, n_feats))

        self.RB11        = nn.ModuleList()
        for i in range(self.num_res_blocks[1]):
            self.RB11.append(ResBlock(in_channels  = n_feats, out_channels = n_feats))
        self.conv11_tail = conv3x3(n_feats, n_feats)

        ### subpixel 1 -> 2
        self.conv12      = conv3x3(n_feats, n_feats*4)
        self.ps12        = nn.PixelShuffle(2)

        ### stage21, 22        
        self.conv22_head = nn.ModuleList()
        for i in range(top_k):
            self.conv22_head.append(conv3x3(128 + n_feats, n_feats))
            
        self.ex12        = CSFI2(n_feats)

        self.RB21        = nn.ModuleList()
        self.RB22        = nn.ModuleList()
        for i in range(self.num_res_blocks[2]):
            self.RB21.append(ResBlock(in_channels  = n_feats, out_channels = n_feats))
            self.RB22.append(ResBlock(in_channels  = n_feats, out_channels = n_feats))

        self.conv21_tail = conv3x3(n_feats, n_feats)
        self.conv22_tail = conv3x3(n_feats, n_feats)

        ### subpixel 2 -> 3
        self.conv23      = conv3x3(n_feats, n_feats*4)
        self.ps23        = nn.PixelShuffle(2)

        ### stage31, 32, 33
        self.conv33_head = nn.ModuleList()
        for i in range(top_k):
            self.conv33_head.append(conv3x3(64 + n_feats, n_feats))

        self.ex123 = CSFI3(n_feats)

        self.RB31 = nn.ModuleList()
        self.RB32 = nn.ModuleList()
        self.RB33 = nn.ModuleList()
        for i in range(self.num_res_blocks[3]):
            self.RB31.append(ResBlock(in_channels=n_feats, out_channels=n_feats))
            self.RB32.append(ResBlock(in_channels=n_feats, out_channels=n_feats))
            self.RB33.append(ResBlock(in_channels=n_feats, out_channels=n_feats))

        self.conv31_tail = conv3x3(n_feats, n_feats)
        self.conv32_tail = conv3x3(n_feats, n_feats)
        self.conv33_tail = conv3x3(n_feats, n_feats)

        self.merge_tail = MergeTail(n_feats)
        
        
    def forward(self, x, S = None, T_lv3 = None, T_lv2 = None, T_lv1 = None):
        ### shallow feature extraction
        x = self.SFE(x)
        
        ### stage11
        x11 = x
        
        ### soft-attention
        f_lv1 = x11
        for i in range(len(T_lv3)) :
            scale_factor = f_lv1.shape[2] // S[i].shape[2]
            S1           = F.interpolate(S[i], scale_factor = scale_factor, mode='bicubic')
            x11_res      = torch.cat((f_lv1, T_lv3[i] * S1), dim = 1)
            x11_res      = self.conv11_head[i](x11_res) * S1
            x11          = x11 + x11_res
        x11_res = x11  
        
        for i in range(self.num_res_blocks[1]):
            x11_res = self.RB11[i](x11_res)
        x11_res = self.conv11_tail(x11_res)
        x11 = x11 + x11_res
        
        ### stage21, 22
        x21     = x11
        x21_res = x21
        x22     = self.conv12(x11)
        x22     = F.relu(self.ps12(x22))
        
        ### soft-attention
        f_lv2 = x22
        for i in range(len(T_lv3)) :
            scale_factor = f_lv2.shape[2] // S[i].shape[2]
            S2           = F.interpolate(S[i], scale_factor = scale_factor, mode='bicubic')
            x22_res      = torch.cat((f_lv2, T_lv2[i] * S2), dim=1)
            x22_res      = self.conv22_head[i](x22_res) * S2
            x22          = x22 + x22_res        
        x22_res = x22

        x21_res, x22_res = self.ex12(x21_res, x22_res)
        

        for i in range(self.num_res_blocks[2]):
            x21_res = self.RB21[i](x21_res)
            x22_res = self.RB22[i](x22_res)

        x21_res = self.conv21_tail(x21_res)
        x22_res = self.conv22_tail(x22_res)
        x21     = x21 + x21_res
        x22     = x22 + x22_res
        
        ### stage31, 32, 33
        x31     = x21
        x31_res = x31
        x32     = x22
        x32_res = x32
        x33     = self.conv23(x22)
        x33     = F.relu(self.ps23(x33))
        
        f_lv3 = x33
        for i in range(len(T_lv3)) :
            scale_factor = f_lv3.shape[2] // S[i].shape[2]
            S3           = F.interpolate(S[i], scale_factor=scale_factor, mode='bicubic')
            x33_res      = torch.cat((f_lv3, T_lv1[i] * S3), dim=1)
            x33_res      = self.conv33_head[i](x33_res) * S3 
            x33          = x33 + x33_res
        x33_res = x33

        x31_res, x32_res, x33_res = self.ex123(x31_res, x32_res, x33_res)

        for i in range(self.num_res_blocks[3]):
            x31_res = self.RB31[i](x31_res)
            x32_res = self.RB32[i](x32_res)
            x33_res = self.RB33[i](x33_res)

        x31_res = self.conv31_tail(x31_res)
        x32_res = self.conv32_tail(x32_res)
        x33_res = self.conv33_tail(x33_res)
        
        x31 = x31 + x31_res
        x32 = x32 + x32_res
        x33 = x33 + x33_res

        x_tt = self.merge_tail(x31, x32, x33)
                
        return x_tt, x33, x32, x31
    
    
    

    
    
