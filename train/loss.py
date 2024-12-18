import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def focal_loss(inputs, targets, alpha=.25, gamma=2) : 
    inputs = F.sigmoid(inputs)       
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
    BCE_EXP = torch.exp(-BCE)
    loss = alpha * (1-BCE_EXP)**gamma * BCE
    return loss 

def BCE_Dice_loss(pred, target, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    return bce * bce_weight + dice * (1 - bce_weight)

def focal_dice_loss(pred, target, focal_weight=0.5, dice_weight=0.5, alpha=0.25, gamma=2, smooth=1.0):
    focal = focal_loss(pred, target, alpha=alpha, gamma=gamma)
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target, smooth=smooth)
    loss = focal * focal_weight + dice * dice_weight
    return loss