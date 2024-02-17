import os
from imageio import imread
from PIL import Image
import numpy as np
import glob
import random
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms as T

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



class RandomRotate(object):
    def __call__(self, sample):
        k1 = np.random.randint(0, 4)
        sample['LR'] = np.rot90(sample['LR'], k1).copy()
        sample['HR'] = np.rot90(sample['HR'], k1).copy()
        sample['LR_sr'] = np.rot90(sample['LR_sr'], k1).copy()
        k2 = np.random.randint(0, 4)
        sample['Ref'] = np.rot90(sample['Ref'], k2).copy()
        sample['Ref_sr'] = np.rot90(sample['Ref_sr'], k2).copy()
        return sample


class RandomFlip(object):
    def __call__(self, sample):
        if (np.random.randint(0, 2) == 1):
            sample['LR'] = np.fliplr(sample['LR']).copy()
            sample['HR'] = np.fliplr(sample['HR']).copy()
            sample['LR_sr'] = np.fliplr(sample['LR_sr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['Ref'] = np.fliplr(sample['Ref']).copy()
            sample['Ref_sr'] = np.fliplr(sample['Ref_sr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['LR'] = np.flipud(sample['LR']).copy()
            sample['HR'] = np.flipud(sample['HR']).copy()
            sample['LR_sr'] = np.flipud(sample['LR_sr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['Ref'] = np.flipud(sample['Ref']).copy()
            sample['Ref_sr'] = np.flipud(sample['Ref_sr']).copy()
        return sample


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


class TrainSet(Dataset):
    def __init__(self, args, transform=transforms.Compose([RandomFlip(), RandomRotate(), ToTensor()]) ):
        self.input_list = sorted([os.path.join(args.dataset_dir, 'train/input', name) for name in 
            os.listdir( os.path.join(args.dataset_dir, 'train/input') )])
        self.ref_list = sorted([os.path.join(args.dataset_dir, 'train/ref', name) for name in 
            os.listdir( os.path.join(args.dataset_dir, 'train/ref') )])
        self.transform = transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        ### HR
        if random.random() < 0.5 :  
            HR      = imread(self.input_list[idx])
            Ref = imread(self.ref_list[idx])
        else :
            Ref = imread(self.input_list[idx])
            HR  = imread(self.ref_list[idx])
            
            
            
        h, w, c = Ref.shape
        if h % 16 != 0 or w % 16 != 0:
            h_new = np.math.ceil(h / 16) * 16
            w_new = np.math.ceil(w / 16) * 16
            Ref = cv2.copyMakeBorder(Ref, 0, h_new - h, 0, w_new - w, cv2.BORDER_REPLICATE)

        h, w, c = HR.shape
        if h % 16 != 0 or w % 16 != 0:
            h_new = np.math.ceil(h / 16) * 16
            w_new = np.math.ceil(w / 16) * 16
            HR = cv2.copyMakeBorder(HR, 0, h_new - h, 0, w_new - w, cv2.BORDER_REPLICATE)
            
        h,w = HR.shape[:2]
        #HR = HR[:h//4*4, :w//4*4, :]

        ### LR and LR_sr
        LR    = np.array(Image.fromarray(HR).resize((w//4, h//4), Image.BICUBIC))
        LR_sr = np.array(Image.fromarray(LR).resize((w, h), Image.BICUBIC))

        ### Ref and Ref_sr
        h2, w2     = Ref.shape[:2]
        Ref_sr = np.array(Image.fromarray(Ref).resize((w2//4, h2//4), Image.BICUBIC))
        Ref_sr = np.array(Image.fromarray(Ref_sr).resize((w2, h2), Image.BICUBIC))
    


        ### change type
        LR     = LR.astype(np.float32)
        LR_sr  = LR_sr.astype(np.float32)
        HR     = HR.astype(np.float32)
        Ref    = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)

        ### rgb range to [-1, 1]
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
        return sample

class TestSet(Dataset):
    def __init__(self, args, ref_level='1', transform=transforms.Compose([ToTensor()])):
        self.input_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'test/CUFED5', '*_0.png')))
        self.ref_list   = sorted(glob.glob(os.path.join(args.dataset_dir, 'test/CUFED5', 
            '*_' + ref_level + '.png')))
        self.transform  = transform
        # Crop image dependig on kernel size
        self.filled     = 4 * args.stride

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        ### HR
        HR   = imread(self.input_list[idx])
        h, w = HR.shape[:2]
        h, w = h // self.filled  * self.filled , w // self.filled  * self.filled 
        HR   = HR[:h, :w, :] ### crop to the multiple of 4

        ### LR and LR_sr
        LR    = np.array(Image.fromarray(HR).resize((w // 4, h // 4), Image.BICUBIC))
        LR_sr = np.array(Image.fromarray(LR).resize((w, h), Image.BICUBIC))

        ### Ref and Ref_sr
        Ref    = imread(self.ref_list[idx])
        h2, w2 = Ref.shape[:2]
        h2, w2 = h2 // self.filled * self.filled , w2 // self.filled * self.filled 
        Ref    = Ref[:h2, :w2, :]
        Ref_sr = np.array(Image.fromarray(Ref).resize((w2 // 4, h2 // 4), Image.BICUBIC))
        Ref_sr = np.array(Image.fromarray(Ref_sr).resize((w2, h2), Image.BICUBIC))
        ### change type
        LR     = LR.astype(np.float32)
        LR_sr  = LR_sr.astype(np.float32)
        HR     = HR.astype(np.float32)
        Ref    = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)

        ### rgb range to [-1, 1]
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
        return sample
    
    
    

    
