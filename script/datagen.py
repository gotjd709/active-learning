from torch.utils.data                       import Dataset
from albumentations                         import Compose, HorizontalFlip, ShiftScaleRotate, VerticalFlip, RandomBrightnessContrast
from config                                 import *
import numpy                                as np
import torch
import cv2


class TensorData(Dataset):
    def __init__(self, zip_list, image_size, augmentation=None):
        self.zip_list  = zip_list
        self.image_size = image_size
        self.augmentation = train_aug() if augmentation else test_aug()

    def __len__(self):
        return len(self.zip_list)

    def __getitem__(self, index):
        img_path, msk_path = self.zip_list[index]
        batch_x = cv2.imread(img_path).astype(np.float32)/255
        batch_y = np.zeros(self.image_size + (1,), dtype='uint8')
        mask = cv2.imread(msk_path, 0)
        batch_y[...,0] = np.where(mask==1, 1, 0)
        sample = self.augmentation(image=batch_x, mask=batch_y)
        x_data, y_data = sample['image'], sample['mask']
        x_data = torch.FloatTensor(x_data)
        x_data = x_data.permute(2,0,1)
        y_data = torch.FloatTensor(y_data)
        y_data = y_data.permute(2,0,1)
        return x_data, y_data

def train_aug():
    ret = Compose(
        [
            HorizontalFlip(),
            ShiftScaleRotate(),
            VerticalFlip(),
            RandomBrightnessContrast(),
        ]
    )
    return ret

def test_aug():
    ret = Compose(
        []
    )
    return ret


def dataloader_setting(train_zip, test_zip):
    # set tensordata
    train_data = TensorData(train_zip, IMAGE_SIZE, augmentation=True)
    test_data = TensorData(test_zip, IMAGE_SIZE)
    # set dataloader
    train_loader = torch.utils.data.DataLoader(
        dataset         = train_data,
        batch_size      = BATCH_SIZE,
        shuffle         = SHUFFLE,
        num_workers     = NUM_WORKER,
        pin_memory      = True,
        drop_last       = True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset         = test_data,
        batch_size      = BATCH_SIZE,
        shuffle         = False,
        num_workers     = NUM_WORKER,
        pin_memory      = True,
        drop_last       = True
    )
    return train_loader, test_loader