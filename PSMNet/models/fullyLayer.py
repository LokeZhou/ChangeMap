from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .submodule import *
import skimage.io
from tensorboardX import SummaryWriter
import torchvision.utils as vutils


#cost_log_dir = 'runs/visual_featuremaps_cost'
#cost_writer = SummaryWriter(log_dir=cost_log_dir)


__imagenet_stats = {'mean': [0.21276975, 0.24060462, 0.21573123],
                   'std': [0.18934937, 0.20215742, 0.2023964]}

def get_normalize_invert(tensor):
    mean = __imagenet_stats['mean']
    std = __imagenet_stats['std']
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, inplanes * 2, 3, 2, 1,1),
                                   nn.ReLU(inplace=True),
                                   )

        self.conv2 = nn.Sequential(convbn(inplanes * 2, inplanes * 2, 3, 1, 1,1)
                                   )

        self.conv3 = nn.Sequential(convbn(inplanes * 2, inplanes * 2, 3, 2, 1,1),
                                   nn.ReLU(inplace=True),
                                   )

        self.conv4 = nn.Sequential(convbn(inplanes * 2, inplanes * 2, 3, 1, 1,1),
                                   nn.ReLU(inplace=True),
                                   )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm2d(inplanes * 2),
            )  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm2d(inplanes),
            )  # +x

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post

class PSMNet(nn.Module):
    def __init__(self, cudaenable):
        super(PSMNet, self).__init__()
        self.cudaenable = cudaenable

        self.feature_extraction = feature_extraction()


        self.dres0 = nn.Sequential(convbn(64, 32, 3, 1, 1,1),
                                   nn.ReLU(inplace=True),
                                   convbn(32, 32, 3, 1, 1,1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn(32, 32, 3, 1, 1,1),
                                   nn.ReLU(inplace=True),
                                   convbn(32, 32, 3, 1, 1,1))

        self.dres2 = hourglass(32)

        self.classif = nn.Sequential(convbn(32, 32, 3, 1, 1,1),
                                     nn.ReLU(inplace=True),
                                     convbn(32, 1, 3, 1, 1,1),
                                     )


        self.sigmoid = nn.Sigmoid()

        self.cbca = nn.AvgPool2d(kernel_size=3, stride=1,padding=1)

        self.max = nn.MaxPool2d(kernel_size=3,stride=3,padding=1)
        print("fullLayer!")

    def forward(self, left, right):

        refimg_fea = self.feature_extraction(left)
        targetimg_fea = self.feature_extraction(right)

        # matching
        if self.cudaenable == True:
            cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1] * 2,
                                              refimg_fea.size()[2], refimg_fea.size()[3]).zero_()).cuda()
        else:
            cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1] * 2,
                                              refimg_fea.size()[2], refimg_fea.size()[3]).zero_())

        Csgm = Variable(torch.FloatTensor(left.size()[0], left.size()[2], left.size()[3]).zero_())


        cost[:, :refimg_fea.size()[1], :, :] = refimg_fea
        cost[:, refimg_fea.size()[1]:, :, :] = targetimg_fea


        cost = cost.contiguous()

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0)

        #out1, pre1, post1 = self.dres2(cost0, None, None)

        cost1 = self.classif(cost0)
        '''x = cost1

        #cost可视化
        img_grid = vutils.make_grid(x, normalize=True, scale_each=True, nrow=2)  # B，C, H, W
        cost_writer.add_image( 'cost_feature_maps', img_grid, global_step=64)

        # 绘制原始图像
        img_raw = get_normalize_invert(left) # 图像去标准化
        img_raw = np.array(img_raw * 255).clip(0, 255).squeeze().astype('uint8')
        cost_writer.add_image('raw img', img_raw, global_step=666)  # j 表示feature map数'''

        fc = nn.Sequential(nn.Linear(refimg_fea.size()[2], refimg_fea.size()[3]),
                           nn.BatchNorm2d(1),
                           nn.ReLU(inplace=True)
                          )

        fc_out = nn.Sequential(nn.Linear(refimg_fea.size()[2], refimg_fea.size()[3]),
                               nn.BatchNorm2d(1)
                               )

        '''pred1 = fc(cost1)
        pred2 = fc(pred1)
        pred3 = fc(pred2)
        pred4 = fc(pred3)

        pred4 = self.sigmoid(pred4)'''

        pred4 = self.sigmoid(cost1)


        similarity_score = F.interpolate(pred4, [left.size()[2], left.size()[3]], mode='bilinear')
        max_cost = self.max(similarity_score)


        cbca1 = self.cbca(similarity_score)
        #cbca1 = F.interpolate(cbca1, [left.size()[2], left.size()[3]], mode='bilinear')

        cbca2 = self.cbca(cbca1)
        #cbca2 = F.interpolate(cbca2, [left.size()[2], left.size()[3]], mode='bilinear')

        cbca3 = self.cbca(cbca2)
        #cbca3 = F.interpolate(cbca3, [left.size()[2], left.size()[3]], mode='bilinear')

        cbca4 = self.cbca(cbca3)
        #cbca4 = F.interpolate(cbca4, [left.size()[2], left.size()[3]], mode='bilinear')

        cbca4 = cbca4 - similarity_score

        for i in range(left.size()[0]):
            Csgm[i,:,:] = self.getSemiglobalMatching(left[i,:,:,:], right[i,:,:,:], similarity_score, cbca4, max_cost)

        out1 = Csgm

        #out2 = self.cbca(out1)
        #out3 = self.cbca(out2)
        out  = self.cbca(out1)

        #out = torch.mean(out, 1)

        print(out)

        return out

    def getSemiglobalMatching(self,left,right,similarity_score,cbca,max_cost):

        w = 2
        kernel_size = 3
        Csgm = Variable(torch.FloatTensor(left.size()[1],left.size()[2]).zero_())

        for k in range(1):
            for i in range(left.size()[1]):
                for j in range(left.size()[2]):
                    cbca_p = cbca[k][0][i][j]
                    #cost_p = similarity_score[k][0][i][j]
                    max_cost_p = max_cost[k][0][i % kernel_size][j % kernel_size]


                    l_i_index,l_j_index = self.getWindown(left.size()[1],left.size()[2],-w,i,j)
                    cost_p_l = similarity_score[k][0][i][l_j_index]
                    max_cost_p_l = max_cost[k][0][i % kernel_size][l_j_index % kernel_size]
                    l_d1,l_d2 = self.computedD(left,right,k,i,j,i,l_j_index)

                    r_i_index, r_j_index = self.getWindown(left.size()[1], left.size()[2], w, i, j)
                    cost_p_r = similarity_score[k][0][i][r_j_index]
                    max_cost_p_r = max_cost[k][0][i % kernel_size][r_j_index % kernel_size]
                    r_d1, r_d2 = self.computedD(left, right, k, i,j,i, r_j_index)

                    t_i_index, t_j_index = self.getWindown(left.size()[1], left.size()[2], -w, i, j)
                    cost_p_t = similarity_score[k][0][t_i_index][j]
                    max_cost_p_t = max_cost[k][0][t_i_index % kernel_size][j % kernel_size]
                    t_d1, t_d2 = self.computedD(left, right, k, i,j,t_i_index, j)

                    b_i_index, b_j_index = self.getWindown(left.size()[1], left.size()[2], w, i, j)
                    cost_p_b = similarity_score[k][0][b_i_index][j]
                    max_cost_p_b = max_cost[k][0][b_i_index % kernel_size][j % kernel_size]
                    b_d1, b_d2 = self.computedD(left, right, k, i,j,b_i_index, j)



                    p1,p2 = self.getParameters(l_d1,l_d2)
                    l_csgm = self.computedSGM(cbca_p,max_cost_p,cost_p_l,max_cost_p_l,p1,p2)

                    p1, p2 = self.getParameters(r_d1, r_d2)
                    r_csgm = self.computedSGM(cbca_p,max_cost_p,cost_p_r,max_cost_p_r,p1, p2)

                    p1, p2 = self.getParameters(t_d1, t_d2)
                    t_csgm = self.computedSGM(cbca_p,max_cost_p,cost_p_t,max_cost_p_t,p1, p2)

                    p1, p2 = self.getParameters(b_d1, b_d2)
                    b_csgm = self.computedSGM(cbca_p, max_cost_p,cost_p_b,max_cost_p_b,p1, p2)
                    Csgm[i][j] = (l_csgm + r_csgm + b_csgm + t_csgm) / 4

        return Csgm



    def getWindown(self,w_max,h_max,w_size,i,j):

        i_index = 0
        j_index = 0

        if (i + w_size) < 0:
            i_index = 0
        elif (i + w_size) > (w_max - 1):
            i_index = w_max - 1
        else:
            i_index = i

        if (j + w_size) < 0:
            j_index = 0
        elif (j + w_size) > (h_max - 1):
            j_index = h_max - 1
        else:
            i_index = j

        return i_index,j_index

    def computedSGM(self,cbca_p,max_cost_p,cost_p_r,max_cost_p_r,p1,p2):

        csgm = cbca_p  + (min(max_cost_p,max_cost_p_r + p1,cost_p_r + p2)
               + max(max_cost_p,max_cost_p_r + p1 ,cost_p_r + p2)) / 2

        return csgm

    def computedD(self,left,right,k,i,j,i_index,j_index):

        d1 = (abs(left[0][i][j] - left[0][i_index][j_index]) + abs(left[1][i][j] - left[1][i_index][j_index])
                + abs(left[2][i][j] - left[2][i_index][j_index])) / 3
        d2 = (abs(right[0][i][j] - right[0][i_index][j_index]) + abs(right[1][i][j] - right[1][i_index][j_index])
                + abs(right[2][i][j] - right[2][i_index][j_index])) / 3

        return d1,d2


    def getParameters(self,d1,d2):

        sgm_p1 = 0
        sgm_p2 = 0
        sgm_d = 0.08
        if (d1 > sgm_d and d2 > sgm_d) or (d1 < sgm_d and d2 < sgm_d):
            sgm_p1 = 0.24
            sgm_p2 = 0.56
        else:
            sgm_p1 = 0.32
            sgm_p2 = 1.28

        return sgm_p1,sgm_p2






