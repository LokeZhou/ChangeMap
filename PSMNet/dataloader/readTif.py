import skimage.io
import numpy as np
from PIL import Image

def readImage(path):
    im = skimage.io.imread(path)
    im = np.array(im)
    return Image.fromarray(im)


def readTrue(path):

    im = skimage.io.imread(path)
    im = np.float32(im)
    im = im / 255.0
    scale = 1

    return im,scale