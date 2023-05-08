import os
import random
import torch
import numpy as np

def partial_label_decomp(labels, task_ids):
    num_organ = 5
    labels = labels.cpu().numpy()
    organ_target = np.zeros(( labels.shape[0], labels.shape[2], labels.shape[3]))  
    for i in range(0,labels.shape[0]):
        temp_target = np.zeros((labels.shape[2],labels.shape[3]))
        temp_target= np.where(labels[i,:,:,:] == int(task_ids[i]), int(task_ids[i]), 0)
        organ_target[i,:,:] = temp_target

    labels= torch.from_numpy(organ_target).cuda()
    return labels

def decomp(labels):
    labels = labels.cpu().numpy()
    new_labels = np.zeros((labels.shape[0],4,labels.shape[1],labels.shape[2]))
    for i in range(4):
        new_labels[:,i,:,:]= np.where(labels==(i+1), 1, 0)
    new_labels= torch.from_numpy(new_labels).cuda()
    return new_labels
    
def get_masked_supervision(labels_w, task_ids, partial_labels,pseudo_labels, separate_mask=False):
    partial_label_hard = partial_label_decomp(labels_w, task_ids)
    partial_label_hard = partial_label_hard.long() #(0,1,2,3,4)
    pseudo_label_t = torch.zeros_like(partial_label_hard)#(4,256,256)
    for b in range(partial_labels.shape[0]):
        for c in range(partial_labels.shape[1]):#0,1,2,3
            if partial_labels[b,c,0,0]==-1: #unlabeled channel
                pseudo_label_t[b:b+1,:,:][pseudo_labels[b:b+1,:,:]==(c+1)]= pseudo_labels[b:b+1,:,:][pseudo_labels[b:b+1,:,:]==(c+1)]
    #labeled_channel
    pseudo_label_t[partial_label_hard>0]=partial_label_hard[partial_label_hard>0] #0,1,2,3,4
    
    if separate_mask:
        mask_labeled = partial_label_hard.clone() #(0,1)/(0,2)/(0,3)/(0,4)
        mask_labeled[mask_labeled==0] = -1
        mask_unlabeled = pseudo_label_t.clone()
        mask_unlabeled[partial_label_hard>0] = -1
        mask_labeled[mask_unlabeled==0]=0 #bg pseudo label
        return pseudo_label_t, partial_label_hard, mask_labeled, mask_unlabeled
    else:
        return pseudo_label_t, partial_label_hard

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