from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
from utils import preprocess 
from models import *
from dataloader import listflowtestfile as DA

# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                    help='select model')
parser.add_argument('--loadmodel', default=None,
                    help='loading model')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--enablecuda',  default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = args.enablecuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


test_left_img, test_right_img = DA.dataloader(args.datapath)


if args.model == 'stackhourglass':
    model = stackhourglass(args.cuda,args.maxdisp)
elif args.model == 'basic':
    model = basic(args.cuda,args.maxdisp)
else:
    print('no model')

if args.cuda:
   model = nn.DataParallel(model, device_ids=[0])
   model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
        model.eval()

        if args.cuda:
           imgL = torch.FloatTensor(imgL).cuda()
           imgR = torch.FloatTensor(imgR).cuda()
        else:
           imgL = torch.FloatTensor(imgL)
           imgR = torch.FloatTensor(imgR)

        imgL, imgR= Variable(imgL), Variable(imgR)

        with torch.no_grad():
            output = model(imgL,imgR)
        output = torch.squeeze(output)
        pred_disp = output.data.cpu().numpy()

        return pred_disp


def main():
   #processed = preprocess.get_transform(augment=False)

   for inx in range(len(test_left_img)):

       imgL_o = (skimage.io.imread(test_left_img[inx]).astype('float32'))
       imgR_o = (skimage.io.imread(test_right_img[inx]).astype('float32'))
       #imgL = processed(imgL_o).numpy()
       #imgR = processed(imgR_o).numpy()
       imgL = imgL_o / 255.0
       imgR = imgR_o / 255.0
       imgL = np.reshape(imgL,[1,3,imgL.shape[0],imgL.shape[1]])
       imgR = np.reshape(imgR,[1,3,imgR.shape[0],imgR.shape[1]])

       # pad to (384, 1248)
       top_pad = 512-imgL.shape[2]
       left_pad = 512-imgL.shape[3]
       imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
       imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

       start_time = time.time()
       pred_disp = test(imgL,imgR)
       print('time = %.2f' %(time.time() - start_time))

       top_pad   = 512-imgL_o.shape[0]
       left_pad  = 512-imgL_o.shape[1]

       if top_pad == 0 and left_pad == 0:
           img = pred_disp
       else:
           img = pred_disp[top_pad:,:-left_pad]
       skimage.io.imsave(test_left_img[inx].split('/')[-1],(img*255).astype('uint16'))

if __name__ == '__main__':
   main()






