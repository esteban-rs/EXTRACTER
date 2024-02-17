import os
from imageio import imread
from PIL import Image
import numpy as np
import glob
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def reject_sample(lst, exception):
    while True:
        choice = np.random.choice(lst)
        if choice != exception:
            return choice

class ToTensor(object):
    def __call__(self, sample):
        LR, LR_sr, HR, Ref, Ref_sr = sample['LR'], sample['LR_sr'], sample['HR'], sample['Ref'], sample['Ref_sr']
        LR = LR.transpose((2,0,1))
        LR_sr = LR_sr.transpose((2,0,1))
        HR = HR.transpose((2,0,1))
        Ref = Ref.transpose((2,0,1))
        Ref_sr = Ref_sr.transpose((2,0,1))
        return {'LR': torch.from_numpy(LR).float(),
                'LR_sr': torch.from_numpy(LR_sr).float(),
                'HR': torch.from_numpy(HR).float(),
                'Ref': torch.from_numpy(Ref).float(),
                'Ref_sr': torch.from_numpy(Ref_sr).float()}


class TestSet(Dataset):
    def __init__(self, args, ref_level='1', transform=transforms.Compose([ToTensor()])):
        self.input_list    = sorted(glob.glob(os.path.join(args.dataset_dir, '*_HR.png')))
        self.lr    = sorted(glob.glob(os.path.join(args.dataset_dir, '*_LR.png')))
        self.lr_sr = sorted(glob.glob(os.path.join(args.dataset_dir, '*_bicubic.png')))

        self.transform  = transform
        self.filled = 4 * args.stride

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        HR   = imread(self.input_list[idx])
        if len(HR.shape) < 3 :
            HR = np.stack([HR, HR, HR], axis = 2)
        h, w = HR.shape[:2]
        h, w = h // self.filled  * self.filled , w // self.filled  * self.filled 
        HR   = HR[:h, :w, :] ### crop to the multiple of 4

        ### LR and LR_sr
        LR    = np.array(Image.fromarray(HR).resize((w // 4, h // 4), Image.BICUBIC))
        LR_sr = np.array(Image.fromarray(LR).resize((w, h), Image.BICUBIC))
                
        filename = reject_sample(self.input_list, self.input_list[idx])

        Ref    = imread(filename)
        if len(Ref.shape) < 3 :
            Ref = np.stack([Ref, Ref, Ref], axis = 2)
        LR_sr = LR_sr[:h, :w, :]
        h2, w2 = Ref.shape[:2]
        h2, w2 = h2 // self.filled  * self.filled , w2 // self.filled  * self.filled 
        Ref    = Ref[:h2, :w2, :]
        
        
        Ref_sr = np.array(Image.fromarray(Ref).resize((w2//4, h2//4), Image.BICUBIC))
        Ref_sr = np.array(Image.fromarray(Ref_sr).resize((w2, h2), Image.BICUBIC))
        '''
        Ref    = LR_sr
        Ref_sr = LR_sr
        '''
        # change type
        LR     = LR.astype(np.float32)
        LR_sr  = LR_sr.astype(np.float32)
        HR     = HR.astype(np.float32)
        Ref    = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)

        # rgb range to [-1, 1]
        LR     = LR / 127.5 - 1.
        LR_sr  = LR_sr / 127.5 - 1.
        HR     = HR / 127.5 - 1.
        Ref    = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.

        sample = {'LR': LR,  
                  'LR_sr': LR_sr,
                  'HR': HR,
                  'Ref': Ref, 
                  'Ref_sr': Ref_sr}

        if self.transform:
            sample = self.transform(sample)
            
            
        sample['name'] = self.input_list[idx]
        return sample
    
    
