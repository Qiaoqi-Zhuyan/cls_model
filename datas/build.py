import torch.utils.data
from timm.data import Mixup, AugMixDataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
from tools.label_trans import label_str2int
from config.model_config import get_config

config = get_config()

class CustomDataset_Origin(Dataset):
    def __init__(self, csv_path, imgs_path, img_size,transforms=None):
        self.csv_path = csv_path
        self.imgs_path = imgs_path
        self.transform = transforms
        self.df = pd.read_csv(self.csv_path)

        self.imgs = []
        self.labels = []

        for idx, row in tqdm(self.df.iterrows()):
            img_id = row['image_id']
            label = row["label"]
            img_file_path = os.path.join(self.imgs_path, str(img_id) + '_thumbnail.png')
            if os.path.isfile(img_file_path):
                img = Image.open(img_file_path).convert('RGB')
                img = img.resize(img_size)
                label_int = label_str2int[f'{label}']
                if self.transform:
                    img = self.transform(img)

                self.imgs.append(img)
                self.labels.append(label_int)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]


class CustomDataset_augsOnly(Dataset):
    def __init__(self, csv_path, augs_imgs_path, augs_imgs_num, img_size,transforms=None):
        self.csv_path = csv_path
        self.augs_imgs_path = augs_imgs_path
        self.transform = transforms
        self.df = pd.read_csv(self.csv_path)
        self.augs_imgs_num = augs_imgs_num

        self.imgs = []
        self.labels = []

        for idx, row in tqdm(self.df.iterrows()):
            img_id = row['image_id']
            label = row["label"]

            for i in range(self.augs_imgs_num):
                for j in range(self.augs_imgs_num):
                    augs_imgs_file = os.path.join(self.augs_imgs_path,
                                                  str(img_id) + "_thumbnail" + f"_trans{j + 1}" + f"_{i + 1}.png")
                    if os.path.isfile(augs_imgs_file):
                        img = Image.open(augs_imgs_file).convert('RGB')
                        img = img.resize(img_size)
                        label_int = label_str2int[f'{label}']
                        if self.transform:
                            img = self.transform(img)

                        self.imgs.append(img)
                        self.labels.append(label_int)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]


class CustomDataset(Dataset):
    def __init__(self, csv_path, origin_imgs_path, augs_imgs_path, augs_imgs_num, img_size, transforms=None):
        self.csv_path = csv_path
        self.origin_imgs_path = origin_imgs_path
        self.augs_imgs_path = augs_imgs_path
        self.transform = transforms
        self.df = pd.read_csv(self.csv_path)
        self.augs_imgs_num = augs_imgs_num

        self.imgs = []
        self.labels = []

        for idx, row in tqdm(self.df.iterrows()):
            img_id = row['image_id']
            label = row["label"]
            origin_imgs_file = os.path.join(self.origin_imgs_path, str(img_id) + '_thumbnail.png')
            if os.path.isfile(origin_imgs_file):
                img = Image.open(origin_imgs_file).convert('RGB')
                img = img.resize(img_size)
                label_int = label_str2int[f'{label}']

                if self.transform:
                    img = self.transform(img)

                self.imgs.append(img)
                self.labels.append(label_int)

            for i in range(self.augs_imgs_num):
                for j in range(self.augs_imgs_num):
                    augs_imgs_file = os.path.join(self.augs_imgs_path,
                                                  str(img_id) + "_thumbnail" + f"_trans{j + 1}" + f"_{i + 1}.png")
                    if os.path.isfile(augs_imgs_file):
                        img = Image.open(augs_imgs_file).convert('RGB')
                        label_int = label_str2int[f'{label}']

                        if self.transform:
                            img = self.transform(img)

                        self.imgs.append(img)
                        self.labels.append(label_int)

        print("Dataset process finish")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]


def build_loader(config):

    train_dataset, val_dataset = build_dataset(config)

    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    val_data_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )


    mixup_fn = None
    if config.use_mixup:
        mixup_fn = Mixup(
            mixup_alpha=config.mixup_alpha,
            cutmix_alpha= config.cutmix_alpha,
            cutmix_minmax=config.cutmix_minmax,
            prob=config.prob,
            switch_prob=config.switch_prob,
            mode=config.mode,
            label_smoothing=config.label_smoothing,
            num_classes=config.num_classes,
        )

    return train_data_loader, val_data_loader, mixup_fn


def build_dataset(config):

    transforms = build_transform(config)

    if config.mode == "use_origin_datasets":
        custom_dataset = CustomDataset_Origin(
            csv_path=config.csv_path,
            imgs_path=config.img_folder,
            img_size=config.img_size,
            transforms=transforms
        )

    elif config.mode == "use_augs_datasets":
        custom_dataset = CustomDataset_augsOnly(
            csv_path=config.csv_path,
            augs_imgs_path=config.aug_imgs_folder,
            augs_imgs_num=config.aug_imgs_num,
            img_size=config.img_size,
            transforms=transforms
        )
    elif config.mode == "use_fuse_datasets":
        custom_dataset = CustomDataset(
            csv_path=config.csv_path,
            origin_imgs_path=config.img_folder,
            augs_imgs_path=config.aug_imgs_folder,
            augs_imgs_num=config.aug_imgs_num,
            img_size=config.img_size,
            transforms=transforms
        )

    train_len = int(len(custom_dataset) * config.split)
    val_len = len(custom_dataset) - train_len

    train_dataset, val_dataset = torch.utils.data.random_split(
        custom_dataset, [train_len, val_len]
    )

    if config.mode.use_augmixDataset:
        train_dataset = AugMixDataset(train_dataset, num_splits=3)

    return train_dataset, val_dataset


'''
    origin dataset: mean: (0.4752056213329383, 0.4150572881620091, 0.4733410321946013), std: (0.4166546716518853, 0.3729801066158773, 0.41309897077716035)
      augs dataset: 
'''

def build_transform(config):

    if config.use_randEraseing:
        transform = transforms.Compose(
            [
                transforms.Resize(config.img_size),
                transforms.RandomErasing(p=0.5),
                transforms.CenterCrop(config.img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=config.mean,
                    std=config.std)
            ]
        )

    else:
        transform = transforms.Compose(
            [
                transforms.Resize(config.img_size),
                transforms.CenterCrop(config.img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=config.mean,
                    std=config.std)
            ]
        )

    return transform







