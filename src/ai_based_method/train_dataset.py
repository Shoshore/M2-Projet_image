from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection
import torch

def collate_only_images(batch):
    images = []
    for item in batch:
        img = item[0]
        if not isinstance(img, torch.Tensor):
            img = transforms.Compose([
                transforms.CenterCrop((256, 256)),
                transforms.ToTensor()
            ])(img)
        images.append(img)
    return torch.stack(images, dim=0)

class Train_dataset():

    def __init__(self):
        self._dataset = CocoDetection(
            root="./data/train2017",
            annFile="./data/annotations/captions_train2017.json",
            transform=None
        )
        self.batch_size = 32
        self.img_size = 256


    def image_preparation(self, is_train: bool = True):
        if is_train:
            return transforms.Compose([
                transforms.RandomCrop((self.img_size, self.img_size), pad_if_needed=True),
                transforms.RandomHorizontalFlip(p = 0.5),
                transforms.ToTensor()
            ])
        return transforms.Compose([
            transforms.CenterCrop((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])



    def test_code(self):
        transform = self.image_preparation(False)
        self._dataset.transform = transform
        return DataLoader(
            self._dataset,
            batch_size = self.batch_size,
            shuffle = False,
            collate_fn = collate_only_images
        )



    def train_dataset(self):
        transform = self.image_preparation()
        self._dataset.transform = transform
        return DataLoader(
            self._dataset,
            batch_size = self.batch_size,
            shuffle = True,
            collate_fn = collate_only_images
        )