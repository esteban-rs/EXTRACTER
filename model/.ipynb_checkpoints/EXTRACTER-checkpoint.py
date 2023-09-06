import torch
import torch.nn as nn
import torch.nn.functional as F
from model import FE, SearchTransfer, CSFI, GDE

class EXTRACTER(nn.Module):
    def __init__(self, args):
        super(EXTRACTER, self).__init__()
        self.args            = args
        self.num_res_blocks  = list( map(int, args.num_res_blocks.split('+')) )
        self.num_grad_blocks = list( map(int, args.num_grad_blocks.split('+')) )
        
        self.FE              = FE.FE(requires_grad = True)
        self.SearchTransfer  = SearchTransfer.SearchTransfer(args)
        self.CSFI            = CSFI.CSFI(num_res_blocks = self.num_res_blocks, 
                                               n_feats  = args.n_feats, 
                                                 top_k  = args.top_k)
        self.GDE             = GDE.GDE(num_grad_blocks  = self.num_grad_blocks,
                                               n_feats  = args.n_feats)

    def forward(self, lr = None, lrsr = None, ref = None, refsr = None):
        _,             _, lrsr_lv3  = self.FE((lrsr  + 1) / 2)
        _,             _, refsr_lv3 = self.FE((refsr + 1) / 2)
        ref_lv1, ref_lv2, ref_lv3   = self.FE((ref   + 1) / 2)
        S, T_lv3, T_lv2, T_lv1      = self.SearchTransfer(lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3)
        x_tt, T3, T2, T1            = self.CSFI(lr, S, T_lv3, T_lv2, T_lv1)
        sr                          = self.GDE(lr, x_tt, T3, T2, T1)
        
        return sr, S

