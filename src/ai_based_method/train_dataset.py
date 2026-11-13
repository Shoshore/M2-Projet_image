from os import listdir
from os.path import join
from PIL.Image import open
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize, Compose, RandomCrop, RandomHorizontalFlip, ToTensor


class _ImagesOnly(Dataset):

    def __init__(self, root):
        self.root = root
        self.files = sorted(
            f for f in listdir(root) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        self.transform = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = open(join(self.root, self.files[idx])).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


class Train_dataset():

    def __init__(self, root = "./src/ai_based_method/data/COCO/val2017"):
        self._dataset = _ImagesOnly(root)
        self.batch_size = 16
        self.img_size = 256

    def image_preparation(self):
        norm = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return Compose([
            RandomCrop((self.img_size, self.img_size), pad_if_needed=True),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            norm
        ])

    def train_dataset(self):
        self._dataset.transform = self.image_preparation()
        return DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )