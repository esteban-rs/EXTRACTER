import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SearchTransfer(nn.Module):
    def __init__(self, args):
        super(SearchTransfer, self).__init__()
        self.search_kernel_size     = args.unfold_kernel_size
        self.stride_search          = args.stride
        self.padding_search         = args.padding
        
        self.unfold_kernel_size_lv3 = self.search_kernel_size
        self.stride_lv3             = self.stride_search
        self.padding_lv3            = self.padding_search
        self.unfold_kernel_size_lv2 = 2 * self.unfold_kernel_size_lv3
        self.stride_lv2             = 2 * self.stride_lv3
        self.padding_lv2            = 2 * self.padding_lv3
        self.unfold_kernel_size_lv1 = 4 * self.unfold_kernel_size_lv3
        self.stride_lv1             = 4 * self.stride_lv3
        self.padding_lv1            = 4 * self.padding_lv3
        self.top_k                  = args.top_k
    
    def bis(self, input, dim, index):
        # batch index select
        # input : [N, ?, ?, ...]
        # dim   : scalar > 0
        # index : [N, idx]
        views        = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse      = list(input.size())
        expanse[0]   = -1
        expanse[dim] = -1
        index        = index.clone().view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def Search(self, lr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3): 
        batch, c,  H, W  = lr_lv3.size()
        
        ## ---- LOW RESOLUTION SPACE ----
        Q  = F.unfold(lr_lv3,   kernel_size = self.search_kernel_size, 
                                    padding = self.padding_search, 
                                     stride = self.stride_search)
        K = F.unfold(refsr_lv3, kernel_size = self.search_kernel_size, 
                                    padding = self.padding_search, 
                                     stride = self.stride_search)
        # Find correlations
        K = K.permute(0, 2, 1)                      # [N, C*k*k,  Hr*Wr] -> [N, Hr*Wr, C*k*k]
        K = F.normalize(K, dim = 2)                 # [N, Hr*Wr, C*k*k]
        Q = F.normalize(Q, dim = 1)                 # [N, C*k*k, H*W]

        R_lv3    = torch.bmm(K, Q) / 12                  # [N, Hr*Wr, H*W]
        S, H_lv3 = torch.max(R_lv3, dim = 1)        # [N, H*W]
        
        # select normalized patches in low-resolution space
        K_ = self.bis(K.permute(0, 2, 1) , 2, H_lv3)  # [N, Hr*Wr, C*k*k] -> [N, C*k*k, H*W]
        K_ = K_.permute(0, 2, 1)                      # [N, Hr*Wr, C*k*k]
        
        ## ---- HIGH RESOLUTION SPACE ----
        V_lv3_unfold = F.unfold(ref_lv3, kernel_size = self.unfold_kernel_size_lv3,
                                             padding = self.padding_lv3, 
                                              stride = self.stride_lv3)
        V_lv2_unfold = F.unfold(ref_lv2, kernel_size = self.unfold_kernel_size_lv2, 
                                             padding = self.padding_lv2, 
                                              stride = self.stride_lv2)
        V_lv1_unfold = F.unfold(ref_lv1, kernel_size = self.unfold_kernel_size_lv1, 
                                             padding = self.padding_lv1, 
                                              stride = self.stride_lv1)
        # select textures
        T_lv3_unfold = self.bis(V_lv3_unfold, 2, H_lv3)
        T_lv2_unfold = self.bis(V_lv2_unfold, 2, H_lv3)
        T_lv1_unfold = self.bis(V_lv1_unfold, 2, H_lv3)

        return Q, K_, T_lv3_unfold, T_lv2_unfold, T_lv1_unfold
    
    
    def Transfer(self, h, w, soft, hard, V_lv3, V_lv2, V_lv1) :
        T_lv3 = []
        T_lv2 = []          
        T_lv1 = []         
        S     = []
        for i in range(hard.shape[1]):            
            # Select texture by index (Hard Attention)
            T_lv3_unfold = self.bis(V_lv3, 2, hard[:, i, :])
            T_lv2_unfold = self.bis(V_lv2, 2, hard[:, i, :])
            T_lv1_unfold = self.bis(V_lv1, 2, hard[:, i, :])

            # Fold sums overlaping elements, must divide by the sum factor
            # For kernel (3, 3) factor is 9
            # For kernel (6, 6) factor is 4
            
            '''
            divisor_lv3 = torch.ones_like(T_lv3_unfold)
            divisor_lv2 = torch.ones_like(T_lv2_unfold)
            divisor_lv1 = torch.ones_like(T_lv1_unfold)            
            
            divisor_lv3 = F.fold(divisor_lv3, output_size = (h, w),                      
                                              kernel_size = self.unfold_kernel_size_lv3, 
                                                  padding = self.padding_lv3, 
                                                   stride = self.stride_lv3)
            
            divisor_lv2 = F.fold(divisor_lv2, output_size =(2 * h, 2 * w),
                                              kernel_size = self.unfold_kernel_size_lv2, 
                                                  padding = self.padding_lv2, 
                                                   stride = self.stride_lv2)
            divisor_lv1 = F.fold(divisor_lv1, output_size = (4 * h, 4 * w), 
                                              kernel_size = self.unfold_kernel_size_lv1,
                                                  padding = self.padding_lv1, 
                                                  stride = self.stride_lv1)
            '''
            
            T_lv3_ = F.fold(T_lv3_unfold, output_size = (h, w),                      
                                          kernel_size = self.unfold_kernel_size_lv3, 
                                              padding = self.padding_lv3, 
                                               stride = self.stride_lv3) / 9
            T_lv2_ = F.fold(T_lv2_unfold, output_size = (2 *h, 2 * w), 
                                          kernel_size = self.unfold_kernel_size_lv2, 
                                              padding = self.padding_lv2, 
                                               stride = self.stride_lv2) / 9
            T_lv1_ = F.fold(T_lv1_unfold, output_size = (4 * h, 4 * w), 
                                          kernel_size = self.unfold_kernel_size_lv1,
                                              padding = self.padding_lv1, 
                                               stride = self.stride_lv1) / 9  
            
            S_      = soft[:, i, :].view(soft[:, i, :].size(0), 1, 
                                         h // self.stride_search , w // self.stride_search)
            
            T_lv3.append(T_lv3_)
            T_lv2.append(T_lv2_)
            T_lv1.append(T_lv1_)
            S.append(S_)

        return S, T_lv3, T_lv2, T_lv1

    def forward(self, lr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3):
        batch, c, h, w = lr_lv3.size()

        Q2, K2, V_lv3, V_lv2, V_lv1 = self.Search(lr_lv3, refsr_lv3, 
                                                  ref_lv1, ref_lv2, ref_lv3)
        # Low Resolution Re-Search
        R_lv3                  = torch.bmm(K2, Q2) # [N, Hr*Wr, H*W]
        # Attention matrices (S_k, H_k)
        S__, H__               = torch.topk(R_lv3, self.top_k, dim = 1, largest=True, sorted=True)
        S, T_lv3, T_lv2, T_lv1 = self.Transfer(h, w, S__, H__, V_lv3, V_lv2, V_lv1)
        
        return S, T_lv3, T_lv2, T_lv1 
    
