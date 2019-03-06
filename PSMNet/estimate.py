from PIL import Image, ImageOps
import numpy as np
import math

def CC(left,right):
    left_x = np.mean(left)
    right_x = np.mean(right)



    l_array = left - left_x
    r_array = right - right_x


    p = (l_array * r_array).sum()

    l_array *= l_array
    r_array *= r_array


    q = l_array.sum() *r_array.sum()
    q = pow(q,0.5)

    p = p/q

    return p

def UIQI(left,right):

    left_x = np.mean(left)
    right_x = np.mean(right)

    l_array = left - left_x
    r_array = right - right_x


    sigmoid_xx = np.mean((l_array*l_array))
    sigmoid_yy = np.mean((r_array*r_array))

    sigmoid_x = pow(sigmoid_xx,0.5)
    sigmoid_y = pow(sigmoid_yy,0.5)

    sigmoid_xy = np.mean((l_array*r_array))

    p = (sigmoid_xy/(sigmoid_x*sigmoid_y)) * (2*left_x*right_x/(pow(left_x,2)+pow(right_x,2))) *(2*sigmoid_x*sigmoid_y/(sigmoid_xx+sigmoid_yy))

    return p

def RMSE(left,right):
    p = left - right
    p *= p

    p = np.sqrt(np.mean(p))

    return p


def RASE(left,right):

    sum = 0
    for i in range(3):
        p = RMSE(left[:,:,i],right[:,:,i])
        p *= p
        sum += p
    sum /= 3
    p = pow(sum,0.5) / 2

    return p

def ERGAS(left,right):

    hl = 100 * left.size / right.size
    sum = 0;
    for i in range(3):
        rmse = RMSE(left[:,:,i],right[:,:,i])
        x = np.mean(left[:,:,i])
        p = rmse / x
        p *= p
        sum += p

    p = hl * pow(sum,0.5)
    return p

def PSNR(left,right):
    max = np.max(left)
    mse = RMSE(left,right)
    mse *= mse
    max = pow(max,2)
    p = 10 * math.log((max / mse))
    return p

def SAM(left,right):
    left_sum = 0
    right_sum = 0
    sum = 0
    for i in range(3):
        left_sum += (left[:,:,i] * left[:,:,i]).sum()
        right_sum += (right[:,:,i] * right[:,:,i]).sum()
        sum += (left[:,:,i] * right[:,:,i]).sum()
    p = sum / (pow(left_sum,0.5) * pow(right_sum,0.5))

    if p > 1:
        p = 1
    if p < 0:
        p = 0
    p = math.acos(p) * 57.32

    return p

if __name__ == '__main__':
    reference_img =  './predata/train/left/2_196.png'
    fuse_img = 'predata/train/right/2_196.png'

    imgL_o = Image.open(reference_img).convert('RGB')
    imgR_o = Image.open(fuse_img).convert('RGB')



    left_img = np.array(imgL_o)
    right_img = np.array(imgR_o)


    print("CC:",CC(left_img,right_img))
    print("UIQI:",UIQI(left_img,right_img))
    print("RMSE:",RMSE(left_img,right_img))
    print("RASE:",RASE(left_img,right_img))
    print("ERGAS:",ERGAS(left_img,right_img))
    print("PSNR:",PSNR(left_img,right_img))
    print("SAM:",SAM(left_img,right_img))