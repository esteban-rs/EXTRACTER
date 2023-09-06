from utils.files import mkExpDir
from dataset import dataloader
from model import EXTRACTER
from loss.loss import get_loss_dict
from trainer import Trainer

import os
import torch
import torch.nn as nn
import warnings
import glob
import time

from argparse import Namespace
from torchinfo import summary
from utils.ploting import plot_results

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    args   = Namespace()
    args.save_dir      = 'extracter_rec'
    args.reset         =  True
    args.log_file_name = 'EXTRACTER.log'
    args.logger_name   = 'EXTRACTER'
    args.dataset       = 'CUFED'                      # Which dataset to train and test
    args.dataset_dir   = r'/home/esteban/Data/CUFED/' # Directory of dataset
    args.num_workers   = 9
    
    # Parameters
    args.img_training_size = 160
    args.num_res_blocks    = '8+16+8+4'  # Residual blocks for features [IFE, lv1, lv2, lv3]
    args.num_grad_blocks   = '8+9+9+9'   # # Residual blocks for features [IFE, lv1, lv2, lv3]
    args.n_feats           = 64          # The number of channels in network
    args.top_k             = 2           # Top K matches for each patch
    args.GAN_k             = 2           # Training discriminator k times when training generator once
    
    ## Loss setings
    args.rec_w             = 1.0         # The weight of reconstruction loss
    args.per_w             = 1e-2        # The weight of perceptual loss
    args.grd_w             = 1e-3        # The weight of transferal perceptual loss
    args.adv_w             = 1e-3        # The weight of adversarial loss
    
    ## Patch Setting and Top Features Setting
    args.unfold_kernel_size = 3
    args.stride             = 1
    args.padding            = 1
    ## Optimizer Settings
    args.beta1              = 0.9        # The beta1 in Adam optimizer
    args.beta2              = 0.999      # The beta2 in Adam optimizer
    args.eps                = 1e-8       # The eps in Adam optimizer
    args.lr_rate            = 1e-4       # Learning rate
    args.lr_rate_dis        = 1e-4       # Learning rate of discriminator
    args.lr_rate_lte        = 1e-5       # Learning rate of LTE
    args.decay              = 1e-4       # Learning rate decay type
    args.gamma              = 0.5        # Learning rate decay factor for step decay
    ## 
    args.batch_size         = 9          # Training batch size
    args.num_init_epochs    = 300        # The number of init epochs which are trained with only reconstruction loss
    args.num_epochs         = 0          # The number of training epochs
    args.print_every        = 600        # Print period
    args.save_every         = 5          # Save period
    args.val_every          = 1          # Validation period
    args.show_every         = 5          # Plot Results

    ## evaluate / test / finetune setting
    args.eval               = True       # Evaluation mode
    args.eval_save_results  = False      # Save each image during evaluation
    args.model_path         = None       # The path of model to evaluation
    args.test               = False      # Test mode
    
    
    _logger     = mkExpDir(args)
    _dataloader = dataloader.get_dataloader(args) if (not args.test) else None
    
    args.cpu    = False
    device      = torch.device('cpu' if args.cpu else 'cuda') # Choose CPU or GPU
    torch.cuda.set_device(0)   
    _model      = EXTRACTER.EXTRACTER(args).to(device)
    _loss_all   = get_loss_dict(args, _logger)
    t           = Trainer(args, _logger, _dataloader, _model, _loss_all)
    summary(_model)

    for epoch in range(1, args.num_init_epochs + 1):
        start = time.time()
        t.train(current_epoch=epoch, is_init=True)
        print('Training time for epoch ',epoch, ': ', time.time() - start,'seconds.')
        if (epoch % args.val_every == 0):
            t.evaluate(current_epoch = epoch)
        if (epoch % args.show_every == 0):
            plot_results(model = t.model, dataloader = _dataloader, total_images = 5, save = True, epoch = epoch, device = device, args = args)
    for epoch in range(1, args.num_epochs + 1):
        start = time.time()   
        t.train(current_epoch = epoch, is_init = False)
        print('Training time for epoch ',epoch, ': ', time.time() - start,'seconds.')
        if (epoch % args.val_every == 0):
            t.evaluate(current_epoch = epoch)
            plot_results(model = t.model, dataloader = _dataloader, total_images = 5, save = True, epoch = epoch, device = device, args = args)
