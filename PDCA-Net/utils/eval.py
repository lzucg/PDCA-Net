import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


from IOUMetric import IOUMetric


def dice_coeff(pred, gt, smooth=1, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d activation function operation")

    #pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = 2 * (intersection + smooth) / (unionset + smooth)
    return loss.sum(), N


def mean_iou_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1)
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum - intersection

    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    return iou


def mean_dice_np(y_true, y_pred, **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    """
    axes = (0, 1)  # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)

    smooth = .001
    dice = 2 * (intersection + smooth) / (mask_sum + smooth)
    return dice


def eval_net(net, loader, device, n_class=1):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if n_class == 1 else torch.long
    tot = 0
    n_val = len(loader) 
    N = 0
    iou_bank = []
    acc_bank = []
    dice_bank = []

    iou_metric = IOUMetric(2)
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)
            mask_pred = net(imgs)

            if n_class > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()

                img = pred.squeeze(0).cpu().numpy().astype(int)

                true_label = true_masks.squeeze(0).cpu().numpy().astype(int)

                iou = mean_iou_np(true_label, img)
                dice = mean_dice_np(true_label, img)
                acc = np.sum(img == true_label) / (img.shape[1] * img.shape[2])

                acc_bank.append(acc)
                iou_bank.append(iou)
                dice_bank.append(dice)

                l, n = dice_coeff(pred, true_masks)
                tot += l
                N += n
            pbar.update()

    return tot / N, np.mean(acc_bank)


