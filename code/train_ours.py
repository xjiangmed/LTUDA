import random
import timeit
import torch
import os
import os.path as osp
import numpy as np
import torch.nn.functional as F
from options.train_options import TrainOptions

from torch.utils.data import DataLoader
from networks.net_factory import net_factory
from dataset.toy_dataset import ToyDataSet
from loss import loss, loss_proto
import medpy.metric.binary as mmb
from tensorboardX import SummaryWriter
from utils.util import truncate,read_lists,seed_torch,get_masked_supervision,decomp
from networks.ema import ModelEMA
from utils.transform import obtain_cutmix_box,mix

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, ep_iter, lr, num_epochs, power):
    lr = lr_poly(lr, ep_iter, num_epochs, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr

def obtain_bbox(batch_size, img_size):
    for i in range(batch_size):  
        if i == 0:
            MixMask = obtain_cutmix_box(img_size).unsqueeze(0)
        else:
            MixMask = torch.cat((MixMask, obtain_cutmix_box(img_size).unsqueeze(0)))
    return MixMask

def main():
    opt = TrainOptions()
    args = opt.initialize()
    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)
    writer = SummaryWriter(args.save_model_path)

    
    # create model
    student_model = net_factory(net_type=args.model) #'unet_proto'
    student_model.to(device)
    
    # load pretrained model in stage1
    print('Loading ', args.reload_path)
    model_dict = student_model.state_dict()
    pretrained_dict = torch.load(args.reload_path) 
    pretrained_dict = {k : v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    student_model.load_state_dict(model_dict)
    
    use_ema = True
    if use_ema:
        ema_model = ModelEMA(student_model, 0.999)
        teacher_model = ema_model.ema
    optimizer = torch.optim.SGD(student_model.parameters(), args.learning_rate, momentum=0.99, nesterov=True)

    # Define the dataset
    train_data = ToyDataSet(args.train_path)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    loss_pBCE = loss.pBCE(num_classes=4).to(device)
    loss_seg_proto = loss_proto.PixelPrototypeCELoss().to(device)
    
    max_dice = 0
    max_dice_ema = 0
    all_tr_loss = []
    iter_num = 0

    for epoch in range(args.num_epochs): 
        start = timeit.default_timer()
        student_model.train()
        teacher_model.train()
        adjust_learning_rate(optimizer, epoch, args.learning_rate, args.num_epochs, args.power)
        epoch_model_loss = []
        train_phar = enumerate(train_loader)

        for iter, batch in train_phar:
            optimizer.zero_grad()
            # labeled and unlabeled: weak view
            images_w, partial_labels_w, volumeName, task_ids, partial_labels_4ch_w = batch[0],batch[1],batch[2],batch[3],batch[4]
            images_w, partial_labels_w, partial_labels_4ch_w = images_w.type(torch.FloatTensor).cuda(), partial_labels_w.type(torch.FloatTensor).cuda(), partial_labels_4ch_w.type(torch.FloatTensor).cuda()
            
            # get pseudo labels from teacher_model for unlabeled data
            outputs_t = teacher_model(images_w) #linear classifier
            probability_t, max_t = torch.max(F.sigmoid(outputs_t), dim=1)
            max_t = max_t+1 
            pseudo_labels = torch.where(probability_t<0.5, torch.as_tensor(0).cuda(), max_t)

            # get masked pseudo labels
            masked_pseudo_labels = get_masked_supervision(partial_labels_w, partial_labels_4ch_w, pseudo_labels)

            # Strong view: Cross-set data augmentation (cutmix) 
            img_size = 256
            bs = images_w.shape[0]
            MixMask = obtain_bbox(bs*2, img_size).cuda()
            images_s1 = mix(MixMask[:bs].unsqueeze(1).repeat(1, 3, 1, 1), images_w, images_w) 
            images_s2 = mix(MixMask[bs:bs*2].unsqueeze(1).repeat(1, 3, 1, 1), images_w, images_w) 
            pseudo_labels_s1 = mix(MixMask[:bs], masked_pseudo_labels, masked_pseudo_labels)
            pseudo_labels_s2 = mix(MixMask[bs:bs*2], masked_pseudo_labels, masked_pseudo_labels)
            pseudo_labels_s1_4ch = decomp(pseudo_labels_s1)
            pseudo_labels_s2_4ch = decomp(pseudo_labels_s2)

            # train student model
            outputs_w = student_model(volume_batch) #linear classifier
            loss_w =  loss_pBCE.forward(outputs_w, partial_labels_4ch_w)
            
            volume_batch = torch.cat([images_s1, images_s2], 0) #linear, L-Prototype and U-Protytpe classifier
            outputs, contrast_logits, contrast_targets = student_model(volume_batch, use_prototype=True, linear_classifier=True, lp_classifier=True, ulp_classifier=True) 
            loss_s1_linear = loss_pBCE.forward(outputs[0][:bs], pseudo_labels_s1_4ch)
            loss_s2_linear = loss_pBCE.forward(outputs[0][bs:bs*2], pseudo_labels_s2_4ch)
            loss_s1_protol = loss_seg_proto(outputs[1][:bs], contrast_logits[0][:bs], contrast_targets[0][:bs], pseudo_labels_s1.long())
            loss_s2_protol = loss_seg_proto(outputs[1][bs:bs*2], contrast_logits[0][bs:bs*2], contrast_targets[0][bs:bs*2], pseudo_labels_s2.long())
            loss_s1_protoul = loss_seg_proto(outputs[2][:bs], contrast_logits[1][:bs], contrast_targets[1][:bs], pseudo_labels_s1.long())
            loss_s2_protoul = loss_seg_proto(outputs[2][bs:bs*2], contrast_logits[1][bs:bs*2], contrast_targets[1][bs:bs*2], pseudo_labels_s2.long())
            
            loss_total = loss_w + (loss_s1_linear + loss_s1_protol + loss_s1_protoul) + (loss_s2_linear+ loss_s2_protol+loss_s2_protoul)

            if use_ema:
                ema_model.update(student_model)
                teacher_model = ema_model.ema
            
            loss_total.backward()
            optimizer.step()
            iter_num += 1
            epoch_model_loss.append(float(loss_total))

            '''
            print('Epoch {}: {}/{}, lr = {:.4}, loss_pBCE = {:.4}'.format( \
                        epoch, iter, len(train_loader), optimizer.param_groups[0]['lr'], loss_total.item(),))   
            '''
        epoch_loss = np.mean(epoch_model_loss)
        end = timeit.default_timer()
        print(end - start, 'seconds')
        all_tr_loss.append(epoch_loss)
        print('Epoch_sum {}: lr = {:.4}, loss_Sum = {:.4}'.format(epoch, optimizer.param_groups[0]['lr'], epoch_loss.item()))
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalars('loss',{'model':epoch_loss.item()}, epoch)

        if (epoch >= 0):
            student_model.eval()
            teacher_model.eval()
            with torch.no_grad():
                val_dice, dice_organs = evaluation(student_model, args.val_path)
                writer.add_scalars('evaluation_dice:', val_dice, epoch)
                print('epoch:'+str(epoch))
                print('dice:'+str(val_dice)+' L '+str(dice_organs[0])+' S '+str(dice_organs[1])+' K '+str(dice_organs[2])+' P'+str(dice_organs[3]))

                if val_dice > max_dice:
                    torch.save(student_model.state_dict(),osp.join(args.save_model_path, 'model_best.pth'))
                    max_dice = val_dice
                    print("=> saved best student_model")
                print("best val dice:{0}".format(max_dice))

                val_dice_ema, dice_organs_ema = evaluation(teacher_model, args.val_path)
                writer.add_scalars('evaluation_dice_ema:', val_dice_ema, epoch)
                print('epoch:'+str(epoch))
                print('dice_ema:'+str(val_dice_ema)+' L '+str(dice_organs_ema[0])+' S '+str(dice_organs_ema[1])+' K '+str(dice_organs_ema[2])+' P'+str(dice_organs_ema[3]))

                if val_dice_ema > max_dice_ema:
                    torch.save(teacher_model.state_dict(),osp.join(args.save_model_path, 'model_best_ema.pth'))
                    max_dice_ema = val_dice_ema
                    print("=> saved best teacher_model")
                print("best val dice_ema:{0}".format(max_dice_ema))

def evaluation(model, data_path):
    test_list = read_lists(data_path)
    dice_list={'L':[],'S':[],'K':[],'P':[]}
    for _, fid in enumerate(test_list):
        _npz_dict = np.load(fid)
        data = _npz_dict['arr_0'].transpose(2,1,0)
        label = _npz_dict['arr_1'].transpose(2,1,0)
        tmp_pred = np.zeros(label.shape)
        for j in range(0,label.shape[0]):
            data_batch = np.zeros([1, 1, 256, 256])
            data_batch[0, 0, :, :] = truncate(data[j, :, :].copy())
            data_bat = torch.from_numpy(data_batch).cuda().float()
            outputs = model(data_bat)
            # threshold-based classifier
            probability_, max_ = torch.max(F.sigmoid(outputs), dim=1)
            max_ = max_+1 
            prediction = torch.where(probability_<0.5, torch.as_tensor(0).cuda(), max_).data.cpu().numpy()
            tmp_pred[j,:,:] = prediction[0,:,:].copy()

        organ_pred = np.zeros(label.shape)
        gt = np.zeros(label.shape)
        for cls in range(1,5):
            organ_pred = np.where(tmp_pred == int(cls), 1, 0)
            gt = np.where(label == int(cls), 1, 0)
            if cls == 1:
                dice_list['L'].append(mmb.dc(organ_pred, gt))
            if cls == 2:
                dice_list['S'].append(mmb.dc(organ_pred, gt))
            if cls == 3:
                dice_list['K'].append(mmb.dc(organ_pred, gt))
            if cls == 4:
                dice_list['P'].append(mmb.dc(organ_pred, gt))
    dice_organs = [np.mean(dice_list['L']), np.mean(dice_list['S']), np.mean(dice_list['K']),np.mean(dice_list['P'])]
    mean_dice = np.mean(dice_organs)
    return mean_dice, dice_organs


        
if __name__ == '__main__':
    seed_torch(seed=43)
    main()
