import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','tif','TIF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):

 #classes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]

 #train = [tr for tr in classes if tr.find('train') > -1]
 #test = [te for te in classes if te.find('test') > -1]

 train_path = filepath + '/' + 'train'
 test_path = filepath + '/' + 'test'

 train_dir =  os.listdir(train_path)
 test_dir = os.listdir(test_path)

 all_left_img = []
 all_right_img = []
 all_left_disp = []
 test_left_img = []
 test_right_img = []
 test_left_disp = []


 for dd in train_dir:
     if dd == 'gt':
       for im in os.listdir(train_path+'/'+dd):
          if is_image_file(train_path+'/'+ dd + im):
             all_left_disp.append(train_path+'/'+dd+'/' + im)
     elif dd == 'left':
       for im in os.listdir(train_path + '/' + dd ):
          if is_image_file(train_path + '/' + dd + im):
              all_left_img.append(train_path + '/' + dd + '/' + im)
     elif dd == 'right':
       for im in os.listdir(train_path + '/' + dd):
          if is_image_file(train_path + '/' + dd + im):
              all_right_img.append(train_path + '/' + dd + '/' + im)


 for dd in test_dir:
     if dd == 'gt':
       for im in os.listdir(test_path+'/'+dd):
          if is_image_file(test_path+'/'+ dd + im):
              test_left_disp.append(test_path+'/'+dd+'/' + im)
     elif dd == 'left':
       for im in os.listdir(test_path + '/' + dd ):
          if is_image_file(test_path + '/' + dd + im):
              test_left_img.append(test_path + '/' + dd + '/' + im)
     elif dd == 'right':
       for im in os.listdir(test_path + '/' + dd):
          if is_image_file(test_path + '/' + dd + im):
              test_right_img.append(test_path + '/' + dd + '/' + im)


 return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp


