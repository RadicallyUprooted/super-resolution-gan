from os import listdir
from os.path import join

from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, Resize, InterpolationMode
from PIL import Image


class DatasetFromFolder(Dataset):
    def __init__(self, dir):
        super(DatasetFromFolder).__init__()
        self.upscale_factor = 4
        self.image_list = [join(dir, x) for x in sorted(listdir(dir))]

    def __getitem__(self, index):
        crop = 24 * self.upscale_factor
        img = Image.open(self.image_list[index])
        img = img.convert('RGB')
        img = RandomCrop(crop)(img)
        
        hr = ToTensor()(img)
        #hr = 2. * hr - 1.
        lr = Compose([Resize(crop // self.upscale_factor, InterpolationMode.BICUBIC),
                    ToTensor()])(img)
            
        return lr, hr

    def __len__(self):
        return len(self.image_list)

