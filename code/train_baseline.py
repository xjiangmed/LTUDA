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
from loss import loss
import medpy.metric.binary as mmb
from tensorboardX import SummaryWriter
from utils.util import truncate,read_lists,seed_torch

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, ep_iter, lr, num_epochs, power):
    lr = lr_poly(lr, ep_iter, num_epochs, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr

def main():
    opt = TrainOptions()
    args = opt.initialize()
    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)
    writer = SummaryWriter(args.save_model_path)

    # create model
    model = net_factory(net_type=args.model)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.99, nesterov=True)
    
    # Define the dataset
    train_data = ToyDataSet(args.train_path)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    loss_pBCE = loss.pBCE(num_classes=4).to(device)
    
    max_dice = 0
    all_tr_loss = []
    iter_num = 0

    for epoch in range(args.num_epochs): 
        start = timeit.default_timer()
        model.train()
        adjust_learning_rate(optimizer, epoch, args.learning_rate, args.num_epochs, args.power)
        epoch_model_loss = []
        train_phar = enumerate(train_loader)

        for iter, batch in train_phar:
            optimizer.zero_grad()
            images_w, partial_labels_w, volumeName, task_ids, partial_labels_4ch_w = batch[0],batch[1],batch[2],batch[3],batch[4]
            images_w, partial_labels_w, partial_labels_4ch_w = images_w.type(torch.FloatTensor).cuda(), partial_labels_w.type(torch.FloatTensor).cuda(), partial_labels_4ch_w.type(torch.FloatTensor).cuda()
            
            outputs = model(images_w)
            loss_total =  loss_pBCE.forward(outputs, partial_labels_4ch_w)#only sup labeled organ
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
            model.eval()
            with torch.no_grad():
                val_dice, dice_organs = evaluation(model, args.val_path)
                writer.add_scalars('evaluation_dice', val_dice, epoch)
                print('epoch:'+str(epoch))
                print('dice:'+str(val_dice)+' L '+str(dice_organs[0])+' S '+str(dice_organs[1])+' K '+str(dice_organs[2])+' P'+str(dice_organs[3]))

                if val_dice > max_dice:
                    torch.save(model.state_dict(),osp.join(args.save_model_path, 'model_best.pth'))
                    max_dice = val_dice
                    print("=> saved best model")
                print("best val dice:{0}".format(max_dice))

def evaluation(model, data_path):
    test_list = read_lists(data_path)
    dice_list={'L':[],'S':[],'K':[],'P':[]}
    for _, fid in enumerate(test_list):
        _npz_dict = np.load(fid)
        data = _npz_dict['arr_0'].transpose(2,1,0)
        label = _npz_dict['arr_1'].transpose(2,1,0)
        tmp_pred = np.zeros(label.shape)
        for j in range(0,label.shape[0]):
            img = truncate(img)
            data_batch = np.zeros([1, 1, 256, 256])
            data_batch[0, 0, :, :] = data[j, :, :].copy()
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
