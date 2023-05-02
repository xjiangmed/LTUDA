import os
import random
import torch
import numpy as np

def decomp(labels):
    labels = labels.cpu().numpy()
    new_labels = np.zeros((labels.shape[0],4,labels.shape[1],labels.shape[2]))
    for i in range(4):
        new_labels[:,i,:,:]= np.where(labels==(i+1), 1, 0)
    new_labels= torch.from_numpy(new_labels).cuda()
    return new_labels

def get_masked_supervision(partial_labels_w, partial_labels_4ch_w, pseudo_label):
    # use partial_GT "partial_labels_w" to replace the labeled organ of "pseudo_label"
    masked_pseudo_label = torch.zeros_like(partial_labels_w)
    for b in range(partial_labels_4ch_w.shape[0]):
        for c in range(partial_labels_4ch_w.shape[1]):
            if partial_labels_4ch_w[b,c,0,0]==-1: #unlabeled organ
                masked_pseudo_label[b:b+1,:,:][pseudo_label[b:b+1,:,:]==(c+1)]= pseudo_label[b:b+1,:,:][pseudo_label[b:b+1,:,:]==(c+1)]
    masked_pseudo_label[partial_labels_w>0]=partial_labels_w[partial_labels_w>0] #0,1,2,3,4
    return masked_pseudo_label

def truncate(CT):
    min_HU = -325
    max_HU = 325
    subtract = 0
    divide = 325.

    # truncate
    CT[np.where(CT <= min_HU)] = min_HU
    CT[np.where(CT >= max_HU)] = max_HU
    CT = CT - subtract
    CT = CT / divide
    return CT  #-1~1


def read_lists(fid):
    with open(fid, 'r') as fd:
        _list = fd.readlines()

    my_list = []
    for _item in _list:
        my_list.append(_item.split('\n')[0])
    return my_list

def seed_torch(seed=43):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False