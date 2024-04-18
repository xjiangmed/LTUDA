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
from loss import loss
from utils.util import truncate,read_lists,seed_torch,get_masked_supervision,decomp

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, lr, num_stemps, power):
    lr = lr_poly(lr, i_iter, num_stemps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr

def main():
    opt = TrainOptions()
    args = opt.initialize()
    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)
    writer = SummaryWriter(args.save_model_path)

    # create model
    student_model = net_factory(net_type=args.model) #'unet'
    student_model.to(device)
    use_ema = True
    if use_ema:
        ema_model = ModelEMA(student_model, 0.999)
        teacher_model = ema_model.ema
    optimizer = torch.optim.SGD(student_model.parameters(), args.learning_rate, momentum=0.99, nesterov=True)

    # Define the dataset
    train_data = ToyDataSet(args.train_path)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    loss_pBCE = loss.pBCE(num_classes=4).to(device)
    
    max_dice = 0
    max_dice_ema = 0
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
            images, labels, volumeName, task_ids, partial_labels = batch[0],batch[1],batch[2],batch[3],batch[4]
            # generate weak and strong views
            aug_images, aug_labels, aug_partial_labels, MixMasks = data_aug(images.detach(), labels.detach(), partial_labels.detach(), task_ids, strong_aug=True, p=1.0, num_strongview=2)
            images_w, labels_w, partial_labels_w = aug_images[0].type(torch.FloatTensor).cuda(), aug_labels[0].type(torch.FloatTensor).cuda(), aug_partial_labels[0].type(torch.FloatTensor).cuda()
            images_s1, labels_s1, partial_labels_s1 = aug_images[1].type(torch.FloatTensor).cuda(), aug_labels[1].type(torch.FloatTensor).cuda(), aug_partial_labels[1].type(torch.FloatTensor).cuda()
            images_s2, labels_s2, partial_labels_s2 = aug_images[2].type(torch.FloatTensor).cuda(), aug_labels[2].type(torch.FloatTensor).cuda(), aug_partial_labels[2].type(torch.FloatTensor).cuda()
            MixMask1, MixMask2 = MixMasks

            # get pseudo labels from teacher_model for unlabeled data
            outputs_t = teacher_model(images_w)
            probability_t, max_t = torch.max(F.sigmoid(outputs_t), dim=1)
            max_t = max_t+1 
            pseudo_labels = torch.where(probability_t<0.5, torch.as_tensor(0).cuda(), max_t)

            # get masked pseudo labels
            masked_pseudo_labels, partial_label_hard = get_masked_supervision(labels_w, task_ids, partial_labels,pseudo_labels)
            
            #apply strong augmentation to the masked pseudo labels
            pseudo_labels_s1 = mix_pseudo_label(masked_pseudo_labels, MixMask1.type(torch.FloatTensor).cuda()).squeeze(1)
            pseudo_labels_s2 = mix_pseudo_label(masked_pseudo_labels, MixMask2.type(torch.FloatTensor).cuda()).squeeze(1)
            pseudo_labels_s1_4ch = decomp(pseudo_labels_s1)
            pseudo_labels_s2_4ch = decomp(pseudo_labels_s2)
            
            # train student model
            bs = images_w.shape[0]
            volume_batch = torch.cat([images_w, images_s1, images_s2], 0)
            outputs = student_model(volume_batch)
            loss_w =  loss_pBCE.forward(outputs[:bs], partial_labels_w)
            #For warmup, the loss weight of labeled organ and unlabeled organ is 2:1
            loss_s1 = loss_pBCE.forward(outputs[bs:bs*2], pseudo_labels_s1_4ch) + loss_pBCE.forward(outputs[bs:bs*2], partial_labels_s1)
            loss_s2 = loss_pBCE.forward(outputs[bs*2:bs*3], pseudo_labels_s2_4ch) + loss_pBCE.forward(outputs[bs*2:bs*3], partial_labels_s2)
            loss_total = loss_w + loss_s1 + loss_s2
            
            if use_ema:
                ema_model.update(student_model)
                teacher_model = ema_model.ema
            
            loss_total.backward()
            optimizer.step()
            epoch_model_loss.append(float(loss_total))
            
            
            # if iter_num %100==0: 
            #     writer.add_image(str(iter_num)+'train/weak_image', vutils.make_grid(images_w.repeat(1, 3, 1, 1).detach(), padding=2, nrow=4, normalize=True),
            #                  iter_num)
            #     image = vis_utils.decode_seg_map_sequence(partial_label_hard.cpu().numpy())
            #     writer.add_image(str(iter_num)+'train/partial_label_hard', vutils.make_grid(image.detach(), padding=2, nrow=4, normalize=False, scale_each=True),
            #                  iter_num)
            #     image = vis_utils.decode_seg_map_sequence(pseudo_labels.cpu().numpy())
            #     writer.add_image(str(iter_num)+'train/pseudo_labels', vutils.make_grid(image.detach(), padding=2, nrow=4, normalize=False, scale_each=True),
            #                  iter_num)
            #     image = vis_utils.decode_seg_map_sequence(masked_pseudo_labels.cpu().numpy())
            #     writer.add_image(str(iter_num)+'train/masked_pseudo_labels', vutils.make_grid(image.detach(), padding=2, nrow=4, normalize=False, scale_each=True),
            #                  iter_num)
            iter_num += 1

            '''
            if (args.local_rank == 0):
                print('Epoch {}: {}/{}, lr = {:.4}, loss_w = {:.4}, loss_s1 = {:.4}, loss_s2 = {:.4}'.format( \
                            epoch, iter, len(train_loader), optimizer.param_groups[0]['lr'], loss_w.item(),
                            loss_s1.item(),loss_s2.item()))   
            '''
        epoch_loss = np.mean(epoch_model_loss)
        end = timeit.default_timer()
        print(end - start, 'seconds')
        print('Epoch_sum {}: lr = {:.4}, loss_Sum = {:.4}'.format(epoch, optimizer.param_groups[0]['lr'], epoch_loss.item()))
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalars('loss',{'model':epoch_loss.item()}, epoch)
        if (epoch >= 0):
            student_model.eval()
            teacher_model.eval()
            with torch.no_grad():
                val_dice, dice_organs = evaluation(student_model, args.val_path)
                print('epoch:'+str(epoch))
                print('dice:'+str(val_dice)+' L '+str(dice_organs[0])+' S '+str(dice_organs[1])+' K '+str(dice_organs[2])+' P'+str(dice_organs[3]))

                if val_dice > max_dice:
                    torch.save(student_model.state_dict(),osp.join(args.save_model_path, 'model_best.pth'))
                    max_dice = val_dice
                    print("=> saved best student_model")
                print("best val dice:{0}".format(max_dice))

                val_dice_ema, dice_organs_ema = evaluation(teacher_model, args.val_path)
                print('epoch:'+str(epoch))
                print('dice_ema:'+str(val_dice_ema)+' L '+str(dice_organs_ema[0])+' S '+str(dice_organs_ema[1])+' K '+str(dice_organs_ema[2])+' P'+str(dice_organs_ema[3]))

                if val_dice_ema > max_dice_ema:
                    torch.save(teacher_model.state_dict(),osp.join(args.save_model_path, 'ema_model_best.pth'))
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
