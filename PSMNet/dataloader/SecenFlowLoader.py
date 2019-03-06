import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import random
from PIL import Image, ImageOps
from .preprocess import *
from .listflowfile import *
from .readTif import *
from .readpfm import *
import numpy as np

import skimage
import skimage.io
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','tif','TIF',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    patharray = np.array([path])
    if patharray[0][-4:] == '.tif' or patharray[0][-4:] == '.TIF':
        return readImage(path)
    else:
        return Image.open(path).convert('RGB')
        #return Image.open(path).convert('L')

def disparity_loader(path):
    patharray = np.array([path])
    if patharray[0][-4:] != '.pfm' or patharray[0][-4:] == '.PFM':
        return readTrue(path)
    else:
        return readPFM(path)


class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, testWeight, testHeight, loader=default_loader, dploader= disparity_loader):
 
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training
        self.testWeight = testWeight
        self.testHeight = testHeight

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]


        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL, scaleL = self.dploader(disp_L)
        dataL = np.ascontiguousarray(dataL,dtype=np.float32)




        if self.training:  
           w, h = left_img.size
           th, tw = 512, 512
           #th, tw = 256,256

 
           x1 = random.randint(0, w - tw)
           y1 = random.randint(0, h - th)


           left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
           right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))




           '''bands = len(left_img.getbands())

           left_img = np.array(left_img)
           right_img = np.array(right_img)


           left_img_tensor = np.zeros((bands,th,tw),dtype = float)
           right_img_tensor = np.zeros((bands, th, tw), dtype=float)


           for i in range(bands):
               if bands == 1:
                   left_img_tensor[i, :, :] = left_img[:, :] / 255.0
                   right_img_tensor[i, :, :] = right_img[:, :] / 255.0
               else:
                   left_img_tensor[i,:,:] = left_img[:,:,i] / 255.0
                   right_img_tensor[i,:,:] = right_img[:,:,i] /255.0



           left_img_tensor = Variable(torch.FloatTensor(left_img_tensor))
           right_img_tensor = Variable(torch.FloatTensor(right_img_tensor))
           '''

           processed = get_transform(augment=False)
           left_img   = processed(left_img)
           right_img  = processed(right_img)


           #return left_img_tensor, right_img_tensor, dataL
           return left_img, right_img, dataL
        else:
           w, h = left_img.size
           left_img = left_img.crop((w-self.testWeight, h-self.testHeight, w, h))
           right_img = right_img.crop((w-self.testWeight, h-self.testHeight, w, h))

           '''bands = len(left_img.getbands())

           left_img = np.array(left_img)
           right_img = np.array(right_img)

           left_img_tensor = np.zeros((bands, self.testHeight, self.testWeight), dtype=float)
           right_img_tensor = np.zeros((bands, self.testHeight, self.testWeight), dtype=float)

           for i in range(bands):
               if bands == 1:
                   left_img_tensor[i, :, :] = left_img[:, :] / 255.0
                   right_img_tensor[i, :, :] = right_img[:, :] / 255.0
               else:
                   left_img_tensor[i,:,:] = left_img[:,:,i] / 255.0
                   right_img_tensor[i,:,:] = right_img[:,:,i] /255.0


           left_img_tensor = Variable(torch.FloatTensor(left_img_tensor))
           right_img_tensor = Variable(torch.FloatTensor(right_img_tensor))
           '''

           processed = get_transform(augment=False)
           left_img       = processed(left_img)
           right_img      = processed(right_img)


           #return left_img_tensor, right_img_tensor, dataL
           return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
