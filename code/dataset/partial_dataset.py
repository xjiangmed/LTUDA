import os
import os.path as osp
import cv2
import torch
import random
import numpy as np
from torch.utils import data
from utils.transform import obtain_cutmix_box, mix

class PartialDataSet(data.Dataset):
    def __init__(self, list_path, return_index=False, aug=True):
        self.list_path = list_path
        self.aug = aug
        with open(self.list_path, 'r') as fp:
            rows = fp.readlines()
        self.img_ids = [row[:-1] for row in rows]
        self.return_index = return_index

    def __len__(self):
        return len(self.img_ids)

    def truncate(self, CT):
        min_HU = -325
        max_HU = 325
        subtract = 0
        divide = 325.

        # truncate
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU
        CT = CT - subtract
        CT = CT / divide
        return CT
    
    def id2trainId_4ch(self, label, task_id):
        # task_id: Liver 1 ; Spleen 2; Kidney 3; Pancreas 4
        shape = label.shape
        results_map = np.zeros((4, shape[1], shape[2])).astype(np.float32)
        for i in range(4):
            if i==int(task_id-1):
                results_map[i, :, :] = np.where(label.squeeze(0)==int(task_id), 1, 0) #labeled organ: 0,1
            else:
                results_map[i, :, :] = results_map[i, :, :] - 1 #unknown organ: -1(ignore)
        return results_map
    
    def id2trainId(self, label, task_id):
        results_map = np.where(label.squeeze(0)==int(task_id), int(task_id), -1) #labeled organ: 1, unknown organ: -1
        return results_map

    
    def __getitem__(self, index):
        image_path = self.img_ids[index].split(' ')[0]
        if image_path[10] == 'x':
            task_id = int(image_path[24])
        else:
            task_id = int(image_path[26])
        name = osp.splitext(osp.basename(image_path))[0]
        npz_dict = np.load(image_path)
        img = npz_dict['arr_0'].transpose(2,1,0) 
        label = npz_dict['arr_1'].transpose(2,1,0) #single-organ (0,1)/(0,2)/(0,3)/(0,4)

        image = self.truncate(img, task_id)
        partial_label = self.id2trainId(label, task_id)
        
        if self.return_index:
            return image, label, name, task_id, partial_label,index
        else:
            return image, label, name, task_id, partial_label

def pad_image(image, h, w, size, padvalue):
    pad_image = image.copy()
    pad_h = max(size[0] - h, 0)
    pad_w = max(size[1] - w, 0)
    if pad_h > 0 or pad_w > 0:
        pad_image = cv2.copyMakeBorder(image, int(pad_h/2), int(pad_h/2), int(pad_w/2), int(pad_w/2),cv2.BORDER_CONSTANT,value=padvalue)
    return pad_image

def data_aug(image, label, partial_label, task_ids, strong_aug=False, p=1.0, num_strongview=1):

    b, d, w, h = image.shape
    data = image.numpy().copy()
    label = label.numpy().copy()
    partial_label = partial_label.numpy().copy()
    size = (1,256,256)
    MixMasks = torch.zeros((num_strongview,) + data.shape)
    for i in range(b):
        rotation = np.random.rand(1)
        scale = np.random.rand(1)
        if scale<=0.5:
            scaler_h = np.random.uniform(0.75, 1.25) #scale
            scale_h = int(size[1] * scaler_h)
            scale_w = scale_h
        
            if scaler_h>1:
                data[i,0,:,:] = cv2.resize(pad_image(data[i,0,:,:], w,h, (scale_h,scale_w), (-1, -1, -1)), (256,256), interpolation=cv2.INTER_AREA)
                label[i,0,:,:] = cv2.resize(pad_image(label[i,0,:,:], w,h, (scale_h,scale_w), (0.0, 0.0, 0.0)), (256,256), interpolation=cv2.INTER_NEAREST)
                partial_label[i,task_ids[i]-1,:,:] = cv2.resize(pad_image(partial_label[i,task_ids[i]-1,:,:], w,h, (scale_h,scale_w), (0.0, 0.0, 0.0)), (256,256), interpolation=cv2.INTER_NEAREST)
            else:
                h_off = random.randint(0, size[1] - scale_h)
                w_off = h_off
                data[i,0,:,:] = cv2.resize(data[i,0, h_off: h_off + scale_h, w_off: w_off + scale_w], (256,256), interpolation=cv2.INTER_AREA)
                label[i,0,:,:] = cv2.resize(label[i,0, h_off: h_off + scale_h, w_off: w_off + scale_w], (256,256), interpolation=cv2.INTER_NEAREST)
                partial_label[i,task_ids[i]-1,:,:] = cv2.resize(partial_label[i,task_ids[i]-1, h_off: h_off + scale_h, w_off: w_off + scale_w], (256,256), interpolation=cv2.INTER_NEAREST)
        
        if rotation <=0.5:
            angle = random.randint(-30, 31) 
            M = cv2.getRotationMatrix2D((size[1]/2,size[2]/2),angle,1)
            label[i,0,:,:] = cv2.warpAffine(label[i,0,:,:], M, (size[1],size[2]), flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0))
            data[i,0,:,:] = cv2.warpAffine(data[i,0,:,:], M, (size[1],size[2]), flags=cv2.INTER_LINEAR, borderValue=(-1, -1, -1))
            partial_label[i,task_ids[i]-1,:,:] = cv2.warpAffine(partial_label[i,task_ids[i]-1,:,:], M, (size[1],size[2]), flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0))
        
        if strong_aug:#strong augmentation: cross-set cutmix
            img_size = size[1] 
            for n in range(num_strongview):
                MixMask = obtain_cutmix_box(img_size, p=p).unsqueeze(0)
                MixMasks[n,i,:,:,:] = MixMask
                
    if strong_aug:
        datas = [0]*(num_strongview+1)
        labels = [0]*(num_strongview+1)
        partial_labels = [0]*(num_strongview+1)
        datas[0], labels[0], partial_labels[0] = torch.from_numpy(data), torch.from_numpy(label), torch.from_numpy(partial_label)
        for n in range(num_strongview):
            datas[n+1], labels[n+1], partial_labels[n+1] = mix(mask = MixMasks[n], data = data, target = label, partial_target=partial_label)
        return datas, labels, partial_labels, MixMasks
    else:
        return torch.from_numpy(data), torch.from_numpy(label), torch.from_numpy(partial_label)

def mix_pseudo_label(pseudo_label, MixMask):
    _, pseudo_label, _ = mix(mask = MixMask, data =None, target = pseudo_label, partial_target=None)
    return pseudo_label
    


