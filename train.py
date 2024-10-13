import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as sm
from tqdm import tqdm
import matplotlib.pyplot as plt

import wandb
from evaluate import evaluate
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss
from utils.utils import get_training_params

dir_img = Path('./data/original/imgs/coronal')
dir_mask = Path('./data/original/masks/coronal')
dir_checkpoint = Path('./checkpoints/')


def train_model(
        model,
        device,
        epochs: int = 15,
        # Mozhi：lets try 4， just for testing
        batch_size: int = 4,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = None,
        imgW: int = None,
        imgH: int = None,
        interval: int = 1,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    # try:
    #     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    # except (AssertionError, RuntimeError, IndexError):
    dataset = BasicDataset(
        images_dir=dir_img,
        mask_dir=dir_mask,
        scale=img_scale if img_scale else None,
        newW=imgW if imgW else None,
        newH=imgH if imgH else None,
        interval=interval if interval else 1  # default interval
    )

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    # MOzhi： try less workers to free more GRAM
    # loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)

    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    if img_scale:
        experiment.config.update(
            dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
        )
    else:
        experiment.config.update(
            dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                val_percent=val_percent, save_checkpoint=save_checkpoint, imgW=imgW, imgH=imgH, amp=amp)
        )
    
    # logging info
    img_info = f"Images scaling:  {img_scale}" if img_scale is not None else f"Image dimensions: Width={imgW}, Height={imgH}"
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        {img_info}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.RMSprop(model.parameters(),
    #                           lr=learning_rate, 
    #                           weight_decay=weight_decay,
    #                           momentum=momentum, 
    #                           foreach=True)

    # goal: maximize Dice score
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # criterion = nn.CrossEntropyLoss()
    criterion = sm.losses.FocalLoss('multiclass')
    global_step = 0
    train_losses = []
    val_losses = []

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_val_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        # Mozhi: this histograms seems to be not ultilized, block it for now to free more GRAM.
                        # histograms = {}
                        # for tag, value in model.named_parameters():
                        #     tag = tag.replace('/', '.')
                        #     if not (torch.isinf(value) | torch.isnan(value)).any():
                        #         histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        #     if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                        #         histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_loss, val_score = evaluate(model, val_loader, device, amp)
                        # print("back_loss: " + str(val_loss))
                        epoch_val_loss += val_loss
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))

        train_losses.append(epoch_loss / len(train_loader))
        val_losses.append(val_loss)
        # print("train_losses_ls: " + str(train_losses))
        # print("val_losses_ls: " + str(val_losses))
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
    
    # save final dice score to file Dice_Scores_Memo_differentInputSize.txt
    with open("Dice_Scores_Memo_S1_differentInputSize.txt", "a") as file:
        file.write(f"{round(val_score, 4)}\n")

    #  print train_losses and val_losses
    epochs_ls = list(range(1, epochs + 1))
    plt.figure()
    plt.plot(epochs_ls, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs_ls, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')

    # scale and img size options
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--scale', '-s', type=float, help='Downscaling factor of the images')
    group.add_argument('--size', '-sz', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'), help='Width and Height of the images')
    
    # old version of scale and img size options
    # parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    # parser.add_argument('--imgW', '-iw', type=int, default=224)
    # parser.add_argument('--imgH', '-ih', type=int, default=224)
    parser.add_argument('--interval', '-itv', type=int, default=1)
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    # model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = sm.Unet('resnet50',
                    encoder_weights='imagenet', 
                    classes=args.classes,
                    activation='softmax')
    model.n_channels = 3
    model.n_classes = args.classes
    model.bilinear = args.bilinear

    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        # Get training parameters
        train_params = get_training_params(args)
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            val_percent=args.val / 100,
            amp=args.amp,
            **train_params  # Unpack the parameters
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()

        # Get training parameters
        train_params = get_training_params(args)
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            val_percent=args.val / 100,
            amp=args.amp,
            **train_params  # Unpack the parameters
        )
