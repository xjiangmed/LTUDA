import numpy as np
from PIL import Image, ImageOps, ImageFilter
import random
import torch
from torchvision import transforms
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy.special import erfinv
import torch.nn.functional as F
from skimage import measure

def obtain_cutmix_box(img_size, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.ones(img_size, img_size)
    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 0

    return mask


def mix(mask, data1, data2): #copy data1 to data2
    # get the random mixing objects
    rand_index = torch.randperm(data2.shape[0])[:data2.shape[0]]
    #Mix
    data = torch.cat([(mask[i] * data2[rand_index[i]] + (1 - mask[i]) * data1[i]).unsqueeze(0) for i in range(data1.shape[0])])
    return data

