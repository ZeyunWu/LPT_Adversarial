# this file contains all functions used for data augmentation on a stochastic basis.
# all inner parameters for some of the augmentation method is uniformly random.
# Zeyun Wu
# March 5th 2021

import numpy as np
import torch
from torchvision import transforms

def crop_and_resize():
    pass

def flip():    
    pass

def rotate():
    pass

def cutout():    
    pass


class Gaussian_noise():
    def __init__(self, mean=0, std=0.02, norm=0.05, p=0.1):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor_img):
        if np.random.rand() > self.p:
            return tensor_img
        noise = torch.zeros_like(tensor_img).normal_(self.mean, self.std)
        return tensor_img.add_(noise)

    def __repr__(self):
        repr = f"{self.__class__.__name__  }(mean={self.mean}, std={self.std}, prob={self.p})"
        return repr
     
        
def inplace_transform(tensor, p=1.0, gn_mean=0, gn_std=0.02):
    '''
    tensor: 3 x 32 x 32 tensor
    '''
    in_transform = transforms.Compose([#transforms.ToPILImage(),
                                       #transforms.RandomHorizontalFlip(p=p),
                                       #transforms.ToTensor(),
                                       Gaussian_noise(mean=gn_mean, std=gn_std, p=p), 
                                       transforms.RandomErasing(p=p, scale=(0.02, 0.2))])
    
    return in_transform(tensor)


class Gaussian_filter():
    def __init__(self, mean=0, std=0.02, norm=0.05, p=0.1):
        self.mean = mean
        self.std = std
        self.p = p
    def __call__(self, tensor_img):
        if np.random.rand() > self.p:
            return tensor_img
        pass

    def __repr__(self):
        repr = f"{self.__class__.__name__  }(mean={self.mean}, std={self.std}, prob={self.p})"
        return repr