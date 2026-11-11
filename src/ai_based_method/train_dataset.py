import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class _ImagesOnly(Dataset):

    def __init__(self, root):
        self.root = root
        self.files = sorted(
            f for f in os.listdir(root) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        self.transform = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.files[idx])).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


class Train_dataset():

    def __init__(self, root = "./src/ai_based_method/data/COCO/val2017"):
        self._dataset = _ImagesOnly(root)
        self.batch_size = 32
        self.img_size = 256

    def image_preparation(self, is_train: bool = True):
        if is_train:
            return transforms.Compose([
                transforms.RandomCrop((self.img_size, self.img_size), pad_if_needed=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])
        return transforms.Compose([
            transforms.CenterCrop((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])

    def test_code(self):
        self._dataset.transform = self.image_preparation(False)
        return DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def train_dataset(self):
        self._dataset.transform = self.image_preparation()
        return DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )