from os import listdir
from os.path import join
from typing import Tuple

from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, Resize, InterpolationMode, ToPILImage
from PIL import Image
import torch

class DatasetFromFolder(Dataset):
    def __init__(self, image_dir: str, crop_size: int, upscale_factor: int):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in sorted(listdir(image_dir))]
        self.hr_crop_size = crop_size * upscale_factor
        
        self.hr_transform = Compose([
            RandomCrop(self.hr_crop_size),
            ToTensor()
        ])
        
        self.lr_transform = Compose([
            ToPILImage(),
            Resize(crop_size, interpolation=InterpolationMode.BICUBIC),
            ToTensor()
        ])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.image_filenames[index]).convert('RGB')
        
        hr_img = self.hr_transform(img)
        lr_img = self.lr_transform(hr_img) # Create LR from the HR cropped image
        
        return lr_img, hr_img

    def __len__(self) -> int:
        return len(self.image_filenames)


