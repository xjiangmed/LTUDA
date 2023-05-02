import torch
import torch.nn.functional as F
import torch.nn as nn

class pBCE(nn.Module):
    def __init__(self, num_classes=4, **kwargs):
        super(pBCE, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        total_loss = []
        for i in range(self.num_classes):
            if i != self.ignore_index:
                ce_loss = self.criterion(predict[:, i], target[:, i]) 
                mask = target[:, i] != -1 #ignore unknown
                ce_loss = ce_loss*mask
                ce_loss = torch.mean(ce_loss, dim=[1,2])
                ce_loss_avg = ce_loss[torch.mean(target[:, i]) != -1].sum() / ce_loss[torch.mean(target[:, i]) != -1].shape[0]
                total_loss.append(ce_loss_avg)
        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss == total_loss]

        return total_loss.sum()/total_loss.shape[0]






















        
