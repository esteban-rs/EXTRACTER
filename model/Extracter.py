import torch
import torch.nn as nn
import torch.nn.functional as F
from model import SR, FE

class Extracter(nn.Module):
    def __init__(self, args):
        super(Extracter, self).__init__()
        self.args            = args
        self.num_res_blocks  = list( map(int, args.num_res_blocks.split('+')) )
        self.num_grad_blocks = list( map(int, args.num_grad_blocks.split('+')) )
        self.top_k           = args.top_k

        self.unfold_kernel_size_lv3 = args.unfold_kernel_size
        self.stride_lv3             = args.stride
        self.padding_lv3            = args.padding
        self.unfold_kernel_size_lv2 = 2 * self.unfold_kernel_size_lv3
        self.stride_lv2             = 2 * self.stride_lv3
        self.padding_lv2            = 2 * self.padding_lv3
        self.unfold_kernel_size_lv1 = 4 * self.unfold_kernel_size_lv3
        self.stride_lv1             = 4 * self.stride_lv3
        self.padding_lv1            = 4 * self.padding_lv3
    

        self.SR              = SR.SR(num_res_blocks  = self.num_res_blocks, 
                                        num_grad_blocks = self.num_grad_blocks,
                                               n_feats  = args.n_feats, 
                                             res_scale  = args.res_scale, 
                                                 top_k  = args.top_k)
        self.FE              = FE.FE(requires_grad = True)
                        
    def forward(self, lr = None, lrsr = None, ref = None, refsr = None):
        _,             _, lrsr_lv3  = self.FE((lrsr  + 1) / 2)
        _,             _, refsr_lv3 = self.FE((refsr + 1) / 2)
        ref_lv1, ref_lv2,   ref_lv3 = self.FE((ref   + 1) / 2)
        
        S, T_lv3, T_lv2, T_lv1      = self.SearchTransfer(lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3)
        sr                          = self.SR(lr, S, T_lv3, T_lv2, T_lv1)
        

        return sr, S, T_lv3, T_lv2, T_lv1

    
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
    
    def SearchPatches(self, lr_f, reflr_f) :
                
        ## ---- LOW RESOLUTION SPACE ----
        Q  = F.unfold(lr_f,   kernel_size = self.unfold_kernel_size_lv3, 
                                  padding = self.stride_lv3, 
                                   stride = self.padding_lv3)
        K = F.unfold(reflr_f, kernel_size = self.unfold_kernel_size_lv3, 
                                  padding = self.stride_lv3, 
                                   stride = self.padding_lv3)
        # Find correlations  
        K = K.permute(0, 2, 1)        # [N, C*k*k,  Hr*Wr] -> [N, Hr*Wr, C*k*k]
        K = F.normalize(K, dim = 2)   # [N, Hr*Wr, C*k*k]
        Q = F.normalize(Q, dim = 1)   # [N, C*k*k, H*W]
        R = torch.bmm(K, Q)           # [N, Hr*Wr, H*W] 
        
        S, H = torch.topk(R, self.top_k, dim = 1, largest=True, sorted=True)
        
        return S, H

    def Transfer(self, h, w, soft, hard, V_lv3, V_lv2, V_lv1) :
        T_lv3 = []
        T_lv2 = []          
        T_lv1 = []         
        S     = []
        
        V_lv3_ = F.unfold(V_lv3, kernel_size = self.unfold_kernel_size_lv3,
                                     padding = self.stride_lv3, 
                                      stride = self.stride_lv3)
        V_lv2_ = F.unfold(V_lv2, kernel_size = self.unfold_kernel_size_lv2, 
                                     padding = self.stride_lv2, 
                                      stride = self.stride_lv2)
        V_lv1_ = F.unfold(V_lv1, kernel_size = self.unfold_kernel_size_lv1, 
                                     padding = self.stride_lv1, 
                                      stride = self.stride_lv1)
        for i in range(hard.shape[1]): 
            # Select texture by index (Hard Attention)
            T_lv3_unfold = self.bis(V_lv3_, 2, hard[:, i, :])
            T_lv2_unfold = self.bis(V_lv2_, 2, hard[:, i, :])
            T_lv1_unfold = self.bis(V_lv1_, 2, hard[:, i, :])

            # Fold sums overlaping elements, must divide by the sum factor
            # For kernel (3, 3) factor is 9
            # For kernel (6, 6) factor is 4
            
            divisor_lv3 = torch.ones_like(T_lv3_unfold)
            divisor_lv2 = torch.ones_like(T_lv2_unfold)
            divisor_lv1 = torch.ones_like(T_lv1_unfold)            
            
            divisor_lv3 = F.fold(divisor_lv3, output_size = (h, w),                      
                                              kernel_size = self.unfold_kernel_size_lv3, 
                                                  padding = self.stride_lv3, 
                                                   stride = self.stride_lv3)
            
            divisor_lv2 = F.fold(divisor_lv2, output_size =(2 * h, 2 * w),
                                              kernel_size = self.unfold_kernel_size_lv2, 
                                                  padding = self.stride_lv2, 
                                                   stride = self.stride_lv2)
            divisor_lv1 = F.fold(divisor_lv1, output_size = (4 * h, 4 * w), 
                                              kernel_size = self.unfold_kernel_size_lv1,
                                                  padding = self.stride_lv1, 
                                                  stride = self.stride_lv1)
            
            T_lv3_ = F.fold(T_lv3_unfold, output_size = (h, w),                      
                                          kernel_size = self.unfold_kernel_size_lv3, 
                                              padding = self.stride_lv3, 
                                               stride = self.stride_lv3) / divisor_lv3
            T_lv2_ = F.fold(T_lv2_unfold, output_size = (2 *h, 2 * w), 
                                          kernel_size = self.unfold_kernel_size_lv2, 
                                              padding = self.stride_lv2, 
                                               stride = self.stride_lv2) / divisor_lv2
            T_lv1_ = F.fold(T_lv1_unfold, output_size = (4 * h, 4 * w), 
                                          kernel_size = self.unfold_kernel_size_lv1,
                                              padding = self.stride_lv1, 
                                               stride = self.stride_lv1) / divisor_lv1  
            
            S_     = soft[:, i, :].view(soft[:, i, :].size(0), 1, h // self.stride_lv3 , w // self.stride_lv3)
            
            T_lv3.append(T_lv3_)
            T_lv2.append(T_lv2_)
            T_lv1.append(T_lv1_)
            S.append(S_)

        return S, T_lv3, T_lv2, T_lv1

    def SearchTransfer(self, lr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3):
        _, _, h, w    = lr_lv3.size()
        S_, H_        = self.SearchPatches(lr_lv3, refsr_lv3)
        S, T3, T2, T1 = self.Transfer(h, w, S_, H_, ref_lv3, ref_lv2, ref_lv1)
        
        
        return S, T3, T2, T1
        