import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        # root:数据集根目录
        # transforms_: 图像变换列表（如缩放、裁剪、归一化）。
        # unaligned: 如果True，A和B图像不配（随机组合，用于无监督学习）；如果False，按索引配对
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))
        # 使用glob扫描root/mode/A/和root/mode/B/下的所有图像文件（支持多种格式，如.jpg、.png）。

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}
    """
    返回一个字典{'A': item_A, 'B': item_B}，其中：
item_A: A域图像(应用transforms)。
item_B: B域图像(应用transforms+如果unaligned=True,随机选；否则对应索引)
"""

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    # 返回A和B文件夹中文件数的最大值，确保数据加载器能遍历所有样本。