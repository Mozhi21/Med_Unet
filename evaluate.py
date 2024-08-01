import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff
import segmentation_models_pytorch as sm

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    iou_score = 0
    fbeta_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # print("mask_true1 size: ", mask_true.size())

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            # print("mask_pred1 size: ", mask_pred.size())

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                # mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                # mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # print("mask_true2 size: ", mask_true.size())
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2)
                # print("mask_true3 size: ", mask_true.size())

                # print("mask_pred2 size: ", mask_pred.size())
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2)
                # print("mask_pred3 size: ", mask_pred.size())

                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                tp, fp, fn, tn = sm.metrics.get_stats(mask_pred[:, 1:], mask_true[:, 1:], mode='multilabel', threshold=0.5)
                fbeta_score += sm.metrics.fbeta_score(tp, fp, fn, tn, beta=0.5, reduction="micro")
                iou_score += sm.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

    net.train()

    # return iou_score / max(num_val_batches, 1)
    # fbeta_score / max(num_val_batches, 1)
    return dice_score / max(num_val_batches, 1)
