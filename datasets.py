import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)  # transform对象
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))
        # 判断数据集是否对齐
        if len(self.files_A) == len(self.files_B):
            self.unaligned = False
        else:
            self.unaligned = True

    def __getitem__(self, index):
        # 转换为RGB色彩空间，防止png等格式报错
        img_A = Image.open(self.files_A[index % len(self.files_A)]).convert('RGB')
        item_A = self.transform(img_A)

        if self.unaligned:
            img_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB')
            item_B = self.transform(img_B)
        else:
            img_B = Image.open(self.files_B[index % len(self.files_B)]).convert('RGB')
            item_B = self.transform(img_B)

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))