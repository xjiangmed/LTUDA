import random
import cv2
import timeit
import torch
import os
import os.path as osp
import numpy as np
import torch.nn.functional as F
from options.train_options import TrainOptions
from torch.utils.data import DataLoader
import medpy.metric.binary as mmb
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from utils import vis_utils
from networks.net_factory import net_factory
from networks.ema import ModelEMA
from dataset.toy_dataset import ToyDataSet, data_aug, mix_pseudo_label
from loss import loss, loss_proto

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, lr, num_stemps, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(lr, i_iter, num_stemps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr

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

def label_decomp(labels):
    labels = labels.cpu().numpy()
    new_labels = np.zeros((labels.shape[0],4,labels.shape[1],labels.shape[2]))
    for i in range(4):
        new_labels[:,i,:,:]= np.where(labels==(i+1), 1, 0)
    new_labels= torch.from_numpy(new_labels).cuda()
    return new_labels

def main():
    opt = TrainOptions()
    args = opt.initialize()
    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)
    writer = SummaryWriter(args.save_model_path)
    use_ema = True
    thresh = 0.5
    
    model1 = net_factory(net_type=args.model) #'unet_proto'
    print('Loading ', args.reload_path)
    model_dict = model1.state_dict()
    pretrained_dict = torch.load(args.reload_path) 
    pretrained_dict = {k : v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model1.load_state_dict(model_dict)
    model1.to(device)
    
    optimizer1 = torch.optim.SGD(model1.parameters(), args.learning_rate, momentum=0.99, nesterov=True)

    if use_ema:
        ema_model = ModelEMA(model1, 0.999)
        model_ema = ema_model.ema

    train_data = ToyDataSet(args.train_path)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    loss_seg_CE = loss.pBCE(num_classes=4).to(device)
    loss_seg_proto = loss_proto.PixelPrototypeCELoss().to(device)
    
    max_dice = 0
    max_dice_ema = 0
    all_tr_loss = []
    all_va_loss = []
    val_best_loss = 999999
    iter_num = 0

    for epoch in range(args.num_epochs): 
        start = timeit.default_timer()
        model1.train()
        model_ema.train()
        adjust_learning_rate(optimizer1, epoch, args.learning_rate, args.num_epochs, args.power)
        epoch_model1_loss = []
        epoch_model2_loss = []
        train_phar = enumerate(train_loader)

        for iter, batch in train_phar:
            optimizer1.zero_grad()
            images, labels, volumeName, task_ids, partial_labels = batch[0],batch[1],batch[2],batch[3],batch[4]
            # generate weak and strong views
            aug_images, aug_labels, aug_partial_labels, MixMasks = data_aug(images.detach(), labels.detach(), partial_labels.detach(), task_ids, strong_aug=True, p=1.0, num_strongview=2)
            images_w, labels_w, partial_labels_w = aug_images[0].type(torch.FloatTensor).cuda(), aug_labels[0].type(torch.FloatTensor).cuda(), aug_partial_labels[0].type(torch.FloatTensor).cuda()
            images_s1, labels_s1, partial_labels_s1 = aug_images[1].type(torch.FloatTensor).cuda(), aug_labels[1].type(torch.FloatTensor).cuda(), aug_partial_labels[1].type(torch.FloatTensor).cuda()
            images_s2, labels_s2, partial_labels_s2 = aug_images[2].type(torch.FloatTensor).cuda(), aug_labels[2].type(torch.FloatTensor).cuda(), aug_partial_labels[2].type(torch.FloatTensor).cuda()
            MixMask1, MixMask2 = MixMasks
            
            preds_w, _  = model_ema(images_w)

            #--------------weak branch-------------------
            preds_w_1, _  = model1(images_w)
            term_seg_BCE_w = loss_seg_CE.forward(preds_w_1, partial_labels_w) #weak: labeled data
            term_model_1 = term_seg_BCE_w
            
            #--------------get pseudo label-------------------
            probability_t, max_t = torch.max(F.sigmoid(preds_w), dim=1)
            max_t = max_t+1 
            max_t = torch.where(probability_t<thresh, torch.as_tensor(0).cuda(), max_t) 
            max_show = max_t

            partial_label_hard = partial_label_decomp(labels_w, task_ids)
            partial_label_hard = partial_label_hard.long() #(0,1,2,3,4)
            pseudo_label_t = torch.zeros_like(partial_label_hard)#(4,256,256)
            for b in range(partial_labels.shape[0]):
                for c in range(partial_labels.shape[1]):#0,1,2,3
                    if partial_labels[b,c,0,0]==-1: #unlabeled channel
                        pseudo_label_t[b:b+1,:,:][max_t[b:b+1,:,:]==(c+1)]= max_t[b:b+1,:,:][max_t[b:b+1,:,:]==(c+1)]
            #labeled_channel
            mask_labeled = partial_label_hard.clone() #(0,1)/(0,2)/(0,3)/(0,4)
            mask_labeled[mask_labeled==0] = -1
            mask_unlabeled = pseudo_label_t.clone()
            mask_unlabeled[partial_label_hard>0] = -1
            mask_labeled[mask_unlabeled==0]=0 #bg pseudo label
            pseudo_label_t[partial_label_hard>0]=partial_label_hard[partial_label_hard>0] #0,1,2,3,4
             
            #apply strong augmentation to the pseudo label
            pseudo_label_t_s1 = mix_pseudo_label(pseudo_label_t, MixMask1.type(torch.FloatTensor).cuda()).squeeze(1) #(0,1,2,3,4)
            pseudo_label_t_s2 = mix_pseudo_label(pseudo_label_t, MixMask2.type(torch.FloatTensor).cuda()).squeeze(1)
            mask_labeled_s1 = mix_pseudo_label(mask_labeled, MixMask1.type(torch.FloatTensor).cuda()) #(0,1,2,3,4)
            mask_labeled_s2 = mix_pseudo_label(mask_labeled, MixMask2.type(torch.FloatTensor).cuda())
            mask_unlabeled_s1 = mix_pseudo_label(mask_unlabeled, MixMask1.type(torch.FloatTensor).cuda()) #(0,1,2,3,4)
            mask_unlabeled_s2 = mix_pseudo_label(mask_unlabeled, MixMask2.type(torch.FloatTensor).cuda())

            # --------------strong branch-------------------
            preds_s1_linear,_ = model1(images_s1, use_prototype = False)#linear predict
            preds_s2_linear,_  = model1(images_s2, use_prototype = False)#linear predict
            
            preds_s1_protol, contrast_logits_s1l, contrast_target_s1l, preds_s1_protoul, contrast_logits_s1ul, contrast_target_s1ul = model1(images_s1, mask_labeled_s1, mask_unlabeled_s1, use_prototype = True)#proto predict
            preds_s2_protol, contrast_logits_s2l, contrast_target_s2l, preds_s2_protoul, contrast_logits_s2ul, contrast_target_s2ul = model1(images_s2, mask_labeled_s2, mask_unlabeled_s2, use_prototype = True)#proto predict

            #unlabeled data: weak to strong1 supervision
            pseudo_label_t_s1_4cha = label_decomp(pseudo_label_t_s1)
            cps_ce_s1_linear = loss_seg_CE.forward(preds_s1_linear, pseudo_label_t_s1_4cha)#linear
            cps_ce_s1_protol = loss_seg_proto(preds_s1_protol, contrast_logits_s1l, contrast_target_s1l, pseudo_label_t_s1.long())
            cps_ce_s1_protoul = loss_seg_proto(preds_s1_protoul, contrast_logits_s1ul, contrast_target_s1ul, pseudo_label_t_s1.long())
            
            pseudo_label_t_s2_4cha = label_decomp(pseudo_label_t_s2)
            cps_ce_s2_linear = loss_seg_CE.forward(preds_s2_linear, pseudo_label_t_s2_4cha)
            cps_ce_s2_protol = loss_seg_proto(preds_s2_protol, contrast_logits_s2l, contrast_target_s2l, pseudo_label_t_s2.long())
            cps_ce_s2_protoul = loss_seg_proto(preds_s2_protoul, contrast_logits_s2ul, contrast_target_s2ul,pseudo_label_t_s2.long())
            
        
            term_model_2 = 1*(cps_ce_s1_linear+ cps_ce_s2_linear) + 1*(cps_ce_s1_protol +  cps_ce_s2_protol) + 1*(cps_ce_s1_protoul +cps_ce_s2_protoul)
                
            
            loss_total = term_model_1 + term_model_2

            if use_ema:
                ema_model.update(model1)
                test_model = model1
                model_ema = ema_model.ema
            else:
                test_model = model1

            loss_total.backward()
            optimizer1.step()

        
            if iter_num %100==0: 
                writer.add_image(str(iter_num)+'train/weak_image', vutils.make_grid(images_w.repeat(1, 3, 1, 1).detach(), padding=2, nrow=4, normalize=True),
                             iter_num)
                image = vis_utils.decode_seg_map_sequence(partial_label_hard.cpu().numpy())
                writer.add_image(str(iter_num)+'train/partial_label_hard', vutils.make_grid(image.detach(), padding=2, nrow=4, normalize=False, scale_each=True),
                             iter_num)
                image = vis_utils.decode_seg_map_sequence(max_show.cpu().numpy())
                writer.add_image(str(iter_num)+'train/max_show', vutils.make_grid(image.detach(), padding=2, nrow=4, normalize=False, scale_each=True),
                             iter_num)
                image = vis_utils.decode_seg_map_sequence(pseudo_label_t.cpu().numpy())
                writer.add_image(str(iter_num)+'train/pseudo_label', vutils.make_grid(image.detach(), padding=2, nrow=4, normalize=False, scale_each=True),
                             iter_num)
                writer.add_image(str(iter_num)+'train/strong_image1', vutils.make_grid(images_s1.repeat(1, 3, 1, 1).detach(), padding=2, nrow=4, normalize=True),
                             iter_num)
                image = vis_utils.decode_seg_map_sequence(pseudo_label_t_s1.cpu().numpy())
                writer.add_image(str(iter_num)+'train/strong_pseudo_label1', vutils.make_grid(image.detach(), padding=2, nrow=4, normalize=False, scale_each=True),
                             iter_num)
                writer.add_image(str(iter_num)+'train/strong_image1', vutils.make_grid(images_s1.repeat(1, 3, 1, 1).detach(), padding=2, nrow=4, normalize=True),
                             iter_num)
                image = vis_utils.decode_seg_map_sequence(pseudo_label_t_s2.cpu().numpy())
                writer.add_image(str(iter_num)+'train/strong_pseudo_label2', vutils.make_grid(image.detach(), padding=2, nrow=4, normalize=False, scale_each=True),
                             iter_num)
            iter_num += 1

            epoch_model1_loss.append(float(term_model_1))
            epoch_model2_loss.append(float(term_model_2))
            
        epoch_model1_loss = np.mean(epoch_model1_loss)
        epoch_model2_loss = np.mean(epoch_model2_loss)
        epoch_loss = epoch_model1_loss+epoch_model2_loss 

        end = timeit.default_timer()
        print(end - start, 'seconds')
        all_tr_loss.append(epoch_loss)
        if (args.local_rank == 0):
            print('Epoch_sum {}: lr = {:.4}, loss_Sum = {:.4}'.format(epoch, optimizer1.param_groups[0]['lr'], epoch_loss.item()))
            writer.add_scalar('learning_rate', optimizer1.param_groups[0]['lr'], epoch)
            writer.add_scalars('loss',{'model1':epoch_model1_loss.item(), 'model2': epoch_model2_loss.item()}, epoch)
            
        if (epoch >= 0) and (args.local_rank == 0):
            test_model.eval()
            model_ema.eval()
            dice_val = []
            with torch.no_grad():
                print('epoch:'+str(epoch))
                dice_final_AbdomenCT_1K, dice_organs_1K = case_validate_multi(test_model, args.val_path_multi_AbdomenCT_1K,test_linear=False)
                print('1K----student model prototype:'+str(dice_final_AbdomenCT_1K)+' L '+str(dice_organs_1K[0])+' S '+str(dice_organs_1K[1])+' K '+str(dice_organs_1K[2])+' P '+str(dice_organs_1K[3]))
                writer.add_scalars('evaluation_dice_final', {'multi_AbdomenCT_1K':dice_final_AbdomenCT_1K}, epoch)
            
                
                if dice_final_AbdomenCT_1K > max_dice:
                    torch.save(model1.state_dict(),osp.join(args.save_model_path, 'model_best.pth'))
                    max_dice = dice_final_AbdomenCT_1K
                    print("=> saved model")


                dice_final_AbdomenCT_1K, dice_organs_1K = case_validate_multi(model_ema, args.val_path_multi_AbdomenCT_1K,test_linear=False)
                print('1K----ema model:'+str(dice_final_AbdomenCT_1K)+' L '+str(dice_organs_1K[0])+' S '+str(dice_organs_1K[1])+' K '+str(dice_organs_1K[2])+' P '+str(dice_organs_1K[3]))
                writer.add_scalars('evaluation_dice_ema', {'multi_AbdomenCT_1K':dice_final_AbdomenCT_1K}, epoch)

                if dice_final_AbdomenCT_1K > max_dice_ema:
                    torch.save(model_ema.state_dict(),osp.join(args.save_model_path, 'ema_model_best.pth'))
                    max_dice_ema = dice_final_AbdomenCT_1K
                    print("=> saved model")
                print("best val dice:{0}".format(max_dice)) 
        
        
    
def read_lists(fid):
    with open(fid, 'r') as fd:
        _list = fd.readlines()

    my_list = []
    for _item in _list:
        my_list.append(_item.split('\n')[0])
    return my_list

def get_prediction(outputs):
    probability_, max_ = torch.max(F.sigmoid(outputs), dim=1)
    max_ = max_+1 
    compact_pred = torch.where(probability_<0.5, torch.as_tensor(0).cuda(), max_)
    compact_pred = compact_pred.data.cpu().numpy()
    out = compact_pred[0,:,:].copy()
    return out

def case_validate_multi(model1, data_path, test_linear=True):

    test_list = read_lists(data_path)
    data_size = [1, 256, 256]
    label_size = [1, 256, 256]

    dice_list={'L1':[],'S1':[],'K1':[],'P1':[],'L2':[],'S2':[],'K2':[],'P2':[],'L':[],'S':[],'K':[],'P':[]}
    organ_number = ['bg','liver','spleen','kidney','pancreas']
    for idx_file, fid in enumerate(test_list):
        # print(idx_file, fid)
        dataid = fid.split('/')[-1].split('.')[0][-3:]#.split('_')[1]
        organ = fid.split('/')[-1].split('_')[0]

        _npz_dict = np.load(fid)
        data = _npz_dict['arr_0'].transpose(2,1,0)
        label = _npz_dict['arr_1'].transpose(2,1,0)
        tmp_pred_ens = np.zeros(label.shape)
        
        for j in range(0,label.shape[0]):
            data_batch = np.zeros([1, data_size[0], data_size[1], data_size[2]])
            label_batch = np.zeros([1, label_size[0], label_size[1], label_size[2]])
            
            img = data[j, :, :].copy()
            mask = label[j, :, :].copy()
            
            img = truncate(img)
            
            data_batch[0, 0, :, :] = img.copy()
            label_batch[0, 0, :, :] = mask.copy()

            data_bat = torch.from_numpy(data_batch).cuda().float()
            label_bat = torch.from_numpy(label_batch).cuda().float()
            
            # linear classifier
            if test_linear:
                outputs, _ = model1(data_bat)
                out = get_prediction(outputs)
            else:
                # proto classifier
                outputs, fea = model1(data_bat, test_proto=True)
                compact_pred = torch.argmax(outputs, dim=1)
                compact_pred = compact_pred.data.cpu().numpy()
                out = compact_pred[0,:,:].copy()
        
            
            tmp_pred_ens[j,:,:] = out.copy()
            
        for cls in range(1,5):
            if cls == 1:
                
                dice_list['L'].append(mmb.dc(tmp_pred_ens== int(cls), label == int(cls)))
            if cls == 2:
                
                dice_list['S'].append(mmb.dc(tmp_pred_ens== int(cls), label == int(cls)))
            if cls == 3:
                
                dice_list['K'].append(mmb.dc(tmp_pred_ens== int(cls), label == int(cls)))
            if cls == 4:
                
                dice_list['P'].append(mmb.dc(tmp_pred_ens== int(cls), label == int(cls)))
       
            
    
    dice_organs = [np.mean(dice_list['L']), np.mean(dice_list['S']), np.mean(dice_list['K']),np.mean(dice_list['P'])]
    dice_final = np.mean(dice_organs)
    return  dice_final, dice_organs
            
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


def seed_torch(seed=42):
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


if __name__ == '__main__':
    seed_torch(seed=43)
    main()
