import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.ndimage as nd



def get_multi_organ_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [0, 0, 128],[128, 128, 0]])


def decode_seg_map_sequence(label_masks, dataset='multi_organ'):
    rgb_masks = []
    #for label_mask in label_masks:
    for i in range(label_masks.shape[0]):
        rgb_mask = decode_segmap(label_masks[i], dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks

def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    """
    
    if dataset == 'multi_organ':
        n_classes = 5
        label_colours = get_multi_organ_labels()
    else:
        raise NotImplementedError

    
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

