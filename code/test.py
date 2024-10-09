import os
import torch
import numpy as np
import torch.nn.functional as F
import medpy.metric.binary as mmb
from options.test_options import TestOptions
from networks.net_factory import net_factory
import SimpleITK as sitk
from skimage.measure import label as LAB
from utils.util import truncate,read_lists


data_size = [1, 256, 256]

def continues_region_extract_organ(label, keep_region_nums):  
    mask = False*np.zeros_like(label)
    regions = np.where(label>=1, np.ones_like(label), np.zeros_like(label))
    L, n = LAB(regions, neighbors=4, background=0, connectivity=2, return_num=True)
    ary_num = np.zeros(shape=(n+1,1))
    for i in range(0, n+1):
        ary_num[i] = np.sum(L==i)
    max_index = np.argsort(-ary_num, axis=0)
    count=1
    for i in range(1, n+1):
        if count<=keep_region_nums: # keep
            mask = np.where(L == max_index[i][0], label, mask)
            count+=1
    label = np.where(mask==True, label, np.zeros_like(label))
    return label

def get_prediction(outputs):
    probability_, max_ = torch.max(F.sigmoid(outputs), dim=1)
    max_ = max_+1 
    compact_pred = torch.where(probability_<0.5, torch.as_tensor(0).cuda(), max_)
    compact_pred = compact_pred.data.cpu().numpy()
    out = compact_pred[0,:,:].copy()
    return out

def test():
    opt = TestOptions()
    args = opt.initialize()
    
    net = net_factory(net_type=args.model) 
    net.load_state_dict(torch.load(args.reload_path))
    print('load model')
    net.eval()
    net.cuda()
    
    test_list = read_lists(args.test_path)
    
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    
    dice_list = {'L':[],'S':[],'K':[],'P':[]}
    hd_list = {'L':[],'S':[],'K':[],'P':[]}
    for idx_file, fid in enumerate(test_list):
        print('processing :', fid)
        dataid = fid.split('.')[0].split('_')[1]
        file_path = os.path.join(args.data_dir, fid)
        print('file_path:', file_path)
        _npz_dict = np.load(file_path)
        data = _npz_dict['arr_0'].transpose(2,1,0)
        label = _npz_dict['arr_1'].transpose(2,1,0)
        tmp_pred = np.zeros(label.shape)
        print('label.shape:', label.shape)
        for j in range(0,label.shape[0]):
            print('j:', j)
            data_batch = np.zeros([1, data_size[0], data_size[1], data_size[2]])
            data_batch[0, 0, :, :] = truncate(data[j, :, :].copy())
            data_bat = torch.from_numpy(data_batch).cuda().float()
            
            if args.linear_classifier:
                outputs, _ = net(data_bat, test_linear=True, test_lproto=False, test_ulproto=False)
                out = get_prediction(outputs) 
            elif args.lp_classifier:
                outputs, _ = net(data_bat, test_linear=False, test_lproto=True, test_ulproto=False)
                compact_pred = torch.argmax(outputs, dim=1)
                compact_pred = compact_pred.data.cpu().numpy()
                out = compact_pred[0,:,:].copy()
            elif args.ulp_classifier:
                outputs, _ = net(data_bat, test_linear=False, test_lproto=False, test_ulproto=True)
                compact_pred = torch.argmax(outputs, dim=1)
                compact_pred = compact_pred.data.cpu().numpy()
                out = compact_pred[0,:,:].copy()
        
            tmp_pred[j,:,:] = out.copy()

        #post
        if args.post:
            pred_liver = (tmp_pred == 1)
            pred_spleen = (tmp_pred == 2)
            pred_kidney = (tmp_pred == 3)
            pred_pancreas = (tmp_pred == 4)
            pred_liver = continues_region_extract_organ(pred_liver, 1)
            pred_spleen = continues_region_extract_organ(pred_spleen, 1)
            pred_kidney = continues_region_extract_organ(pred_kidney, 2)
            pred_pancreas = continues_region_extract_organ(pred_pancreas, 1)

            seg_pred = np.zeros_like(tmp_pred)
            seg_pred = np.where(pred_liver == 1, 1, seg_pred)
            seg_pred = np.where(pred_spleen == 1, 2, seg_pred)
            seg_pred = np.where(pred_kidney == 1, 3, seg_pred)
            seg_pred = np.where(pred_pancreas == 1, 4, seg_pred)
            tmp_pred = seg_pred

        organ_pred = np.zeros(label.shape)
        gt = np.zeros(label.shape)
        test_for_4class = True
        if test_for_4class: # evaluate four organs
            case_dice = 0
            case_hd = 0
            for cls in range(1,5):
                organ_pred = np.where(tmp_pred == int(cls), 1, 0)
                gt = np.where(label == int(cls), 1, 0)
                boud_z, _, _ = np.where(gt == 1)
                bbx_z_min = boud_z.min()
                bbx_z_max = boud_z.max()
                hd_dis = 0
                skip_slices = 0
                for z in range(bbx_z_min,bbx_z_max + 1):
                    if organ_pred[z,:,:].sum() > 0:
                        hd_dis += mmb.hd(organ_pred[z,:,:], gt[z,:,:])
                    else:
                        skip_slices += 1
                hd = hd_dis/(bbx_z_max - bbx_z_min + 1-skip_slices)
                case_dice += mmb.dc(organ_pred, gt)
                case_hd += hd
                
                if cls == 1:
                    dice_list['L'].append(mmb.dc(organ_pred, gt))
                    hd_list['L'].append(hd)
                if cls == 2:
                    dice_list['S'].append(mmb.dc(organ_pred, gt))
                    hd_list['S'].append(hd)
                if cls == 3:
                    dice_list['K'].append(mmb.dc(organ_pred, gt))
                    hd_list['K'].append(hd)
                if cls == 4:
                    dice_list['P'].append(mmb.dc(organ_pred, gt))
                    hd_list['P'].append(hd)
            
        else: # evaluate single organ
            if fid[10] == 'x':
                task_id = int(fid[24])
            else:
                task_id = int(fid[26])
            organ_index = task_id+1 
            organ_pred = np.where(tmp_pred == organ_index, 1, 0)
            boud_z, _, _ = np.where(label == 1)
            bbx_z_min = boud_z.min()
            bbx_z_max = boud_z.max()
            hd_dis = 0
            skip_slices = 0
            for z in range(bbx_z_min,bbx_z_max + 1):
                if organ_pred[z,:,:].sum() > 0:
                    hd_dis += mmb.hd(organ_pred[z,:,:], label[z,:,:])
                else:
                    skip_slices += 1
            hd = hd_dis/(bbx_z_max - bbx_z_min + 1-skip_slices)
            if organ_index == 1:
                dice_list['L'].append(mmb.dc(organ_pred, label))
                hd_list['L'].append(hd)
            if organ_index == 2:
                dice_list['S'].append(mmb.dc(organ_pred, label))
                hd_list['S'].append(hd)
            if organ_index == 3:
                dice_list['K'].append(mmb.dc(organ_pred, label))
                hd_list['K'].append(hd)
            if organ_index == 4:
                dice_list['P'].append(mmb.dc(organ_pred, label))
                hd_list['P'].append(hd)

        new_nii = sitk.GetImageFromArray(tmp_pred.astype(np.uint8))
        sitk.WriteImage(new_nii, args.result_path + '_' + str(dataid) + '.nii.gz')
        
    dice_organs = [np.mean(dice_list['L']), np.mean(dice_list['S']), np.mean(dice_list['K']),np.mean(dice_list['P'])]
    mean_dice = np.mean(dice_organs)

    hd_organs = [np.mean(hd_list['L']), np.mean(hd_list['S']), np.mean(hd_list['K']),np.mean(hd_list['P'])]
    mean_hd = np.mean(hd_organs)

    print ('Dice:')
    print ('liver :%.4f, %.4f' % (np.mean(dice_list['L']),np.std(dice_list['L'])))
    print ('spleen :%.4f, %.4f' % (np.mean(dice_list['S']),np.std(dice_list['S'])))
    print ('kidney :%.4f, %.4f' % (np.mean(dice_list['K']),np.std(dice_list['K'])))
    print ('pancreas :%.4f, %.4f' % (np.mean(dice_list['P']),np.std(dice_list['P'])))
    print ('mean :%.4f' % (mean_dice))
    
    print ('HD:')
    print ('liver :%.4f, %.4f' % (np.mean(hd_list['L']),np.std(hd_list['L'])))
    print ('spleen :%.4f, %.4f' % (np.mean(hd_list['S']),np.std(hd_list['S'])))
    print ('kidney :%.4f, %.4f' % (np.mean(hd_list['K']),np.std(hd_list['K'])))
    print ('pancreas :%.4f, %.4f' % (np.mean(hd_list['P']),np.std(hd_list['P'])))
    print ('mean :%.4f' % (mean_hd))
    
if __name__ == '__main__':
    test()
