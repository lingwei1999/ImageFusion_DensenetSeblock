import cv2
import torch
from kornia.utils import image_to_tensor
from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
import random
import os


class TrainData(Dataset):
    """
    Loading fusion data from hard disk.
    """

    def __init__(self, folder: Path, mode='train', transforms=lambda x: x):
        super(TrainData, self).__init__()

        assert mode in ['val', 'train'], 'mode should be "val" or "train"'

        names = (folder / f'{mode}.txt').read_text().splitlines()
        self.samples = [{
            'name': name,
            'ir': folder / f'infrared/{mode}' /f'{name}', ## change file extension here
            'vi': folder / f'visible/{mode}' /f'{name}', 
        } for name in names]
        
        self.transforms = transforms


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        ir = self.imread(sample['ir'])
        vi = self.imread(sample['vi'])

        ir = self.transforms(ir)
        vi = self.transforms(vi)

        sample = {'name': sample['name'], 'img1': ir, 'img2': vi}

        return sample

    @staticmethod
    def imread(path: Path) -> Tensor:
        img_n = cv2.imread(os.path.abspath(str(path)))
        img_t = image_to_tensor(img_n / 255.).float()
        return img_t


class ValData(Dataset):
    """
    Loading fusion data from hard disk.
    """

    def __init__(self, folder: Path, mode='train', transforms=lambda x: x):
        super(ValData, self).__init__()

        assert mode in ['val', 'train'], 'mode should be "val" or "train"'
        names = (folder / f'{mode}.txt').read_text().splitlines()
        self.samples = [{
            'name': name,
            'ir': folder / f'infrared/{mode}' /f'{name}', ## change file extension here
            'vi': folder / f'visible/{mode}' /f'{name}', 
        } for name in names]

        self.transforms = transforms


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        ir = self.imread(sample['ir'])
        vi = self.imread(sample['vi'])

        ir = self.transforms(ir)
        vi = self.transforms(vi)

        sample = {'name': sample['name'], 'img1': ir, 'img2': vi}

        return sample

    @staticmethod
    def imread(path: Path) -> Tensor:
        img_n = cv2.imread(str(path))
        img_t = image_to_tensor(img_n / 255.).float()
        return img_t


if __name__ == '__main__':
    fd = FusionData(folder=Path('../datasets/FLIR_ADAS_v2'))
    s = fd[0]
    print(s)
