import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=True)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1     = conv3x3(in_channels, out_channels, stride)
        self.relu      = nn.ReLU(inplace=True)
        self.conv2     = conv3x3(out_channels, out_channels)
        
    def forward(self, x):
        x1  = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out

class APFM1(nn.Module):
    def __init__(self,n_feats, top_k, num_res_blocks):
        super(APFM1, self).__init__()
        self.n_feats        = n_feats
        self.top_k          = top_k
        self.num_res_blocks = num_res_blocks
        
        self.conv11_head = nn.ModuleList()
        for i in range(top_k):
            self.conv11_head.append(conv3x3(256 + n_feats, n_feats))
            
        self.F_f = nn.Sequential(
            nn.Linear(self.n_feats, 4 * self.n_feats),
            nn.ReLU(),
            nn.Linear(4 * self.n_feats, self.n_feats),
            nn.Sigmoid()
        )

        self.RB11        = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RB11.append(ResBlock(in_channels  = n_feats, 
                                      out_channels = n_feats))
            
        # out channel: 160
        self.F_p = nn.Sequential(
            conv1x1(self.n_feats, 4 * self.n_feats),
            conv1x1(4 *  self.n_feats, self.n_feats)
        )

     # condense layer
        self.condense = conv3x3(self.n_feats, self.n_feats)

    def forward(self, x, x1, S):
        ### soft-attention
        cor   = x
        for i in range(len(x1)) :
            scale_factor = x.shape[2] // S[i].shape[2]
            S1           = F.interpolate(S[i], scale_factor = scale_factor, mode='bicubic')
            cor_res      = torch.cat((x, x1[i] * S1), dim = 1)
            cor_res      = self.conv11_head[i](cor_res)
            cor          = cor + cor_res

        w = F.adaptive_avg_pool2d(cor, (1, 1)).squeeze()  # (n,c) : (4, 160)
        if len(w.shape) == 1:
            w = w.unsqueeze(dim=0)
        w = self.F_f(w)
        w = w.reshape(*w.shape, 1, 1)
        
        for i in range(self.num_res_blocks):
            cor = self.RB11[i](cor)
        
        cor = self.F_p(cor)
        cor1 = self.condense(w * cor)

        return cor1
    
    
class APFM2(nn.Module):
    def __init__(self,n_feats, top_k, num_res_blocks):
        super(APFM2, self).__init__()
        self.n_feats        = n_feats
        self.top_k          = top_k
        self.num_res_blocks = num_res_blocks

        
        self.conv22_head = nn.ModuleList()
        for i in range(top_k):
            self.conv22_head.append(conv3x3(128 + n_feats, n_feats))
            
        self.F_f = nn.Sequential(
            nn.Linear(self.n_feats, 4 *  self.n_feats),
            nn.ReLU(),
            nn.Linear(4 * self.n_feats,  self.n_feats),
            nn.Sigmoid()
        )

        self.RB22        = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RB22.append(ResBlock(in_channels  = n_feats, 
                                      out_channels = n_feats))
            
        # out channel: 160
        self.F_p = nn.Sequential(
            conv1x1(self.n_feats, 4 * self.n_feats),
            conv1x1(4 * self.n_feats, self.n_feats)
        )
        # condense layer
        self.condense = conv3x3(self.n_feats,  self.n_feats)

    def forward(self, x, x2,S):
        x = F.interpolate(x, scale_factor=2, mode='bicubic')#

        ### soft-attention
        cor   = x
        for i in range(len(x2)) :
            scale_factor = x.shape[2] // S[i].shape[2]
            S2           = F.interpolate(S[i], scale_factor = scale_factor, mode='bicubic')
            cor_res      = torch.cat((x, x2[i] * S2), dim = 1)
            cor_res      = self.conv22_head[i](cor_res)
            cor          = cor + cor_res        
        

        w = F.adaptive_avg_pool2d(cor, (1, 1)).squeeze()  # (n,c) : (4, 160)
        if len(w.shape) == 1:
            w = w.unsqueeze(dim=0)
        w = self.F_f(w)
        w = w.reshape(*w.shape, 1, 1)
        
        for i in range(self.num_res_blocks):
            cor = self.RB22[i](cor)
           
        cor = self.F_p(cor)
        cor2 = self.condense(w * cor)

        return cor2

class APFM3(nn.Module):
    def __init__(self,n_feats, top_k,  num_res_blocks):
        super(APFM3, self).__init__()
        self.n_feats        = n_feats
        self.top_k          = top_k
        self.num_res_blocks = num_res_blocks

        self.conv33_head = nn.ModuleList()
        for i in range(top_k):
            self.conv33_head.append(conv3x3(64 + n_feats, n_feats))
            
        self.F_f = nn.Sequential(
            nn.Linear(self.n_feats, 4 * self.n_feats),
            nn.ReLU(),
            nn.Linear(4 * self.n_feats, self.n_feats),
            nn.Sigmoid()
        )

        self.RB33        = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RB33.append(ResBlock(in_channels  = n_feats, 
                                      out_channels = n_feats))
                        
            
        # out channel: 160
        self.F_p = nn.Sequential(
            conv1x1(self.n_feats, 4 * self.n_feats),
            conv1x1(4 *  self.n_feats, self.n_feats)
        )
        # condense layer
        self.condense = conv3x3(self.n_feats, self.n_feats)

    def forward(self, x, x3,S):
        x = F.interpolate(x, scale_factor=2, mode='bicubic')#

        cor   = x
        for i in range(len(x3)) :
            scale_factor = x.shape[2] // S[i].shape[2]
            S3           = F.interpolate(S[i], scale_factor = scale_factor, mode='bicubic')
            cor_res      = torch.cat((x, x3[i] * S3), dim = 1)
            cor_res      = self.conv33_head[i](cor_res)
            cor          = cor + cor_res        

        w = F.adaptive_avg_pool2d(cor, (1, 1)).squeeze()  # (n,c) : (4, 160)
        if len(w.shape) == 1:
            w = w.unsqueeze(dim=0)
        w = self.F_f(w)
        w = w.reshape(*w.shape, 1, 1)

        for i in range(self.num_res_blocks):
            cor = self.RB33[i](cor)

        cor = self.F_p(cor)
        cor3 = self.condense(w * cor)

        return cor3
    
       

