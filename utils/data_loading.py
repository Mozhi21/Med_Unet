import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as F
import os


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_idx = idx.replace('img_', 'mask_')
    mask_file = list(mask_dir.glob(mask_idx + mask_suffix + '.*'))[0]
    # mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float=None, newW: int=None, newH: int=None, interval: int=1, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.scale = scale
        self.newW = newW
        self.newH = newH
        self.mask_suffix = mask_suffix

        # deal with skipping steps
        self.ids = []
        files = os.listdir(images_dir)
        for i in range(len(files)):
            file = files[i]
            # Full path to the file
            file_path = os.path.join(images_dir, file)
            
            # Check if it's a valid file and meets the conditions
            if os.path.isfile(file_path) and not file.startswith('.'):
                if i % interval == 0:
                    self.ids.append(os.path.splitext(file)[0])

        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, is_mask, scale: float=None, newW: int=None, newH: int=None):
        w, h = pil_img.size
        if scale:
            newW, newH = int(scale * w), int(scale * h)
        elif not newW or not newH:
            raise ValueError("Either scale or both newW and newH must be provided.")
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        # Compute padding to make dimensions divisible by 32
        pad_w = (32 - newW % 32) % 32
        pad_h = (32 - newH % 32) % 32
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        padding = (pad_left, pad_right, pad_top, pad_bottom)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i
            
            mask = torch.from_numpy(mask)
            mask = F.pad(mask, padding, mode='constant', value=0)

            return mask.numpy()

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0
            img = torch.from_numpy(img).float()
            img = F.pad(img, padding, mode='constant', value=0)

            return img.numpy()

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_name = name.replace('img_', 'mask_')

        mask_file = list(self.mask_dir.glob(mask_name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {mask_name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(mask_values=self.mask_values, pil_img = img, is_mask=False, 
                              scale = self.scale if self.scale else None,
                              newW = self.newW if self.newW else None,
                              newH = self.newH if self.newH else None)
        mask = self.preprocess(mask_values=self.mask_values, pil_img=mask, is_mask=True, 
                              scale = self.scale if self.scale else None,
                              newW = self.newW if self.newW else None,
                              newH = self.newH if self.newH else None)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }    

class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')