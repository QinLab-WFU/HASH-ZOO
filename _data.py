import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MyDataset(Dataset):
    """
    Common dataset.

    Args
        root(str): Path of image files.
        img_txt(str): Path of txt file containing image file name & image label.
        transform(callable, optional): Transform images.
    """

    def __init__(self, root, img_txt, transform=None):
        self.root = root
        self.transform = transform

        img_txt_path = os.path.join(root, img_txt)

        # Read files
        image_list = open(img_txt_path, "r").readlines()
        self.imgs = [
            (x.split()[0], np.array(x.split()[1:], dtype=np.float32))
            for x in image_list
        ]

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(open(os.path.join(self.root, path), "rb")).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)

    def get_onehot_targets(self):
        # return np.vstack([x[1] for x in self.imgs])
        return torch.from_numpy(np.vstack([x[1] for x in self.imgs]))


def build_loader(root, data_name, batch_size, num_workers, trans_train=None, trans_test=None, drop_last_train=False,
                 drop_last_test=False, pin_memory=False):
    """load dataset"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if trans_train is None:
        trans_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    if trans_test is None:
        trans_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    if data_name == "cifar":
        data_path = os.path.join(root, "CIFAR-10")
        topk = -1
        num_classes = 10
    elif data_name == "flickr":
        data_path = os.path.join(root, "MIRFLICKR-25K")
        topk = -1
        num_classes = 38
    elif data_name == "coco":
        data_path = os.path.join(root, "MS-COCO")
        topk = -1
        num_classes = 80
    elif data_name == "nuswide":
        data_path = os.path.join(root, "NUS-WIDE-TC21")
        topk = 5000
        num_classes = 21
    else:
        raise NotImplementedError(f"unknown dataset: {data_name}")

    dset_train = MyDataset(data_path, "train.txt", transform=trans_train)
    dset_test = MyDataset(data_path, "test.txt", transform=trans_test)
    dset_database = MyDataset(data_path, "database.txt", transform=trans_test)
    print(f'train set: {len(dset_train)}')
    print(f'query set: {len(dset_test)}')
    print(f'retrieve set: {len(dset_database)}')

    train_loader = DataLoader(
        dset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=drop_last_train,
        pin_memory=pin_memory
    )
    database_loader = DataLoader(
        dset_database, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=drop_last_test,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        dset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=drop_last_test,
        pin_memory=pin_memory
    )

    return train_loader, test_loader, database_loader, topk, num_classes


if __name__ == "__main__":
    train_loader, test_loader, database_loader, topk, num_classes = build_loader("/home/sxz/Downloads/datasets",
                                                                                 "cifar",
                                                                                 128, 4)
    for x in test_loader:
        print(x[0], x[1])
