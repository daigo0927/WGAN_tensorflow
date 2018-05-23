from torch.utils.data import Dataset
from pathlib import Path
from itertools import islice
import numpy as np
import imageio
import torch
import random
import cv2
from functools import partial
from abc import abstractmethod, ABCMeta


class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw),:]


class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def __call__(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]
                                                                                                

class BaseDataset(Dataset, metaclass = ABCMeta):
    def __init__(self, dataset_dir, cropper = 'random', crop_shape = None,
                 resize_shape = None, resize_scale = None):
        self.cropper = cropper
        self.crop_shape = crop_shape
        self.resize_shape = resize_shape
        self.resize_scale = resize_scale

        self.dataset_dir = dataset_dir
        self.find_images()
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = imageio.imread(img_path)

        if self.crop_shape is not None:
            cropper = StaticRandomCrop(img.shape[:2], self.crop_shape) if self.cropper == 'random'\
                      else StaticCenterCrop(img.shape[:2], self.crop_shape)
            img = cropper(img)

        if self.resize_shape is not None:
            resizer = partial(cv2.resize, dsize = (0,0), dst = self.resize_shape)
            img = resizer(img)
            
        elif self.resize_scale is not None:
            resizer = partial(cv2.resize, dsize = (0,0), fx = self.resize_scale, fy = self.resize_scale)
            img = resizer(img)

        return np.array(img)

    @abstractmethod 
    def find_images(self): ...

        
class CelebA(BaseDataset):
    def __init__(self, dataset_dir, cropper = 'random', crop_shape = None,
                 resize_shape = None, resize_scale = None):
        super().__init__(dataset_dir, cropper, crop_shape,
                         resize_shape, resize_scale)
        
    def find_images(self):
        p = Path(self.dataset_dir)
        p_image = p / 'image'
        self.samples = list(map(str, p_image.glob('*.jpg')))
        print(f'Found {len(self.samples)} samples.')


def get_dataset(dataset_name):
    return {'CelebA': CelebA}[dataset_name]
