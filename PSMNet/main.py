from __future__ import print_function
import argparse
import os
import random
import ast
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
from models import *
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/SceneFlowData/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default= None,
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--enablecuda', type=ast.literal_eval, default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--testheight', type=int, default=512, metavar='S',
                    help='test sample pixed height (default: 512)')
parser.add_argument('--testweight', type=int, default=512, metavar='S',
                    help='test sample pixed weight (default: 512)')
parser.add_argument('--trainbatchsize', type=int, default=12, metavar='S',
                    help='train batch size (default: 12)')
parser.add_argument('--testbatchsize', type=int, default=8, metavar='S',
                    help='test batch size (default: 8)')
args = parser.parse_args()
enablecuda =  torch.cuda.is_available() and args.enablecuda


if enablecuda:
    torch.cuda.manual_seed(args.seed)
else:
    torch.manual_seed(args.seed)

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(args.datapath)

TrainImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(all_left_img,all_right_img,all_left_disp, True,args.testweight,args.testheight),
         batch_size= args.trainbatchsize, shuffle= True, num_workers= 8, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(test_left_img,test_right_img,test_left_disp, False,args.testweight,args.testheight),
         batch_size= args.testbatchsize, shuffle= False, num_workers= 4, drop_last=False)


if args.model == 'stackhourglass':
    model = stackhourglass(enablecuda,args.maxdisp)
elif args.model == 'basic':
    model = basic(enablecuda,args.maxdisp)
elif args.model =='fullyLayer':
    model = fullyLayer(enablecuda)
elif args.model == 'graphLayer':
    model = graphLayer(enablecuda)
else:
    print('no model')

if enablecuda:
    model = nn.DataParallel(model)
    model.cuda()


if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

def train(imgL,imgR, disp_L):
        model.train()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))   
        disp_L = Variable(torch.FloatTensor(disp_L))

        disp_true = disp_L
        if enablecuda:
            imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()


       #---------
        mask = disp_true < args.maxdisp
        mask.detach_()
        #----
        optimizer.zero_grad()
        
        if args.model == 'stackhourglass':
            output1, output2, output3 = model(imgL,imgR)
            output1 = torch.squeeze(output1,1)
            output2 = torch.squeeze(output2,1)
            output3 = torch.squeeze(output3,1)
            loss = 0.3 * F.smooth_l1_loss(output1, disp_true, size_average=True) + 0.3 * F.smooth_l1_loss(
                output2, disp_true, size_average=True) + 0.3*F.smooth_l1_loss(output3, disp_true, size_average=True)
        elif args.model == 'basic':
            output3 = model(imgL,imgR)
            output = torch.squeeze(output3,1)
            #loss = F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True)
            loss = F.smooth_l1_loss(output3, disp_true, size_average=True)
        elif args.model == 'fullyLayer':
            output3 = model(imgL, imgR)
            output = torch.squeeze(output3, 1)
            # loss = F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True)
            loss = F.smooth_l1_loss(output3, disp_true, size_average=True)
        elif args.model == 'graphLayer':
            output1, output2, output3 = model(imgL, imgR)
            output1 = torch.squeeze(output1, 1)
            output2 = torch.squeeze(output2, 1)
            output3 = torch.squeeze(output3, 1)
            # loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True)
            loss = 0.3 * F.smooth_l1_loss(output1, disp_true, size_average=True) + 0.3 * F.smooth_l1_loss(
                output2, disp_true, size_average=True) + 0.4 * F.smooth_l1_loss(output3, disp_true, size_average=True)

        loss.backward()
        optimizer.step()

        #return loss.data[0]
        return loss.item()

def test(imgL,imgR,disp_true):
        model.eval()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))   
        if enablecuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()

        #---------
        #mask = disp_true < 192
        #----

        with torch.no_grad():
            output3 = model(imgL,imgR)

        #output = torch.squeeze(output3.data.cpu(),1)[:,4:,:]
        output = torch.squeeze(output3.data.cpu(), 1)[:, :, :]

        if len(disp_true)==0:
           loss = 0
        else:
           #loss = torch.mean(torch.abs(output[mask]-disp_true[mask]))  # end-point-error
           loss = torch.mean(torch.abs(output - disp_true))  # end-point-error
           #loss = torch.sum(torch.abs(output - disp_true))
        return loss

def adjust_learning_rate(optimizer, epoch):
    lr = 0.001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():

    start_full_time = time.time()
    writer = SummaryWriter('runs')
    min_Loss = 10000.0
    min_epoch = 0;
    for epoch in range(1, args.epochs+1):
       print('This is %d-th epoch' %(epoch))
       total_train_loss = 0
       adjust_learning_rate(optimizer,epoch)

       ## training ##
       for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
           start_time = time.time()
           loss = train(imgL_crop,imgR_crop, disp_crop_L)
           print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
           total_train_loss += loss
       print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))
       writer.add_scalar('train_loss',total_train_loss/len(TrainImgLoader),epoch)
       if (total_train_loss/len(TrainImgLoader)) < min_Loss:
           min_Loss = total_train_loss/len(TrainImgLoader)
           min_epoch = epoch

       #SAVE
       savefilename = args.savemodel+'/checkpoint_'+str(epoch)+'.tar'
       torch.save({
		    'epoch': epoch,
		    'state_dict': model.state_dict(),
                    'train_loss': total_train_loss/len(TrainImgLoader),
		}, savefilename)

    print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))
    writer.close()

    #------------- TEST ------------------------------------------------------------
    '''total_test_loss = 0
    for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
        test_loss = test(imgL,imgR, disp_L)
        print('Iter %d test loss = %.3f' %(batch_idx, test_loss))
        total_test_loss += test_loss

    print('total test loss = %.3f' %(total_test_loss/len(TestImgLoader)))
    #----------------------------------------------------------------------------------
	#SAVE test information
    savefilename = args.savemodel+'testinformation.tar'
    torch.save({
		    'test_loss': total_test_loss/len(TestImgLoader),
		}, savefilename)
    '''
    print('min_loss epoch = %d' %(min_epoch))

if __name__ == '__main__':
   main()
    
