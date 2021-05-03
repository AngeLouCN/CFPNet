import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss2d(nn.Module):
    '''
    This file defines a cross entropy loss for 2D images
    '''

    def __init__(self, weight=None, ignore_label=255):
        '''
        :param weight: 1D weight vector to deal with the class-imbalance
        Obtaining log-probabilities in a neural network is easily achieved by adding a LogSoftmax layer in the last layer of your network. 
        You may use CrossEntropyLoss instead, if you prefer not to add an extra layer.
        '''
        super().__init__()

        # self.loss = nn.NLLLoss2d(weight, ignore_index=255)
        self.loss = nn.NLLLoss2d(weight, ignore_index=ignore_label)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, 1), targets)


class FocalLoss2d(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss


class OhemCrossEntropy2dTensor(nn.Module):
    def __init__(self, ignore_label, reduction='elementwise_mean', thresh=0.6, min_kept=256,
                 down_ratio=1, use_weight=True,weight=None):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        self.weight=weight
        if use_weight:
            # weight = torch.FloatTensor(
            #     [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
            #      0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
            #      1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       weight=weight,
                                                       ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(1 - valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(1 - valid_mask, self.ignore_label)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


class CriterionDABNet(nn.Module):
    """
    ICNet loss
    """

    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, reduce=True):
        super(CriterionDABNet, self).__init__()
        self.ignore_index = ignore_index
        self.criterion1 = OhemCrossEntropy2dTensor(ignore_index, thresh=thresh, min_kept=min_kept)
        self.criterion2 = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)

        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds, size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion1(scale_pred, target)

        # scale_pred = F.upsample(input=preds, size=(h, w), mode='bilinear', align_corners=True)
        # loss2 = self.criterion2(scale_pred, target)

        return 0.7*loss1