import os
import torch
import torchvision
import cv2

import numpy as np

from torch.utils.data import Dataset
from PIL import Image
from functools import reduce


class PipelineFacesDatasetGenerator(Dataset):
    negative_extensions = ['.txt']

    def folder_filter(self, _path):
        return reduce(
            lambda x, y: x and y,
            [ext not in _path for ext in self.negative_extensions]
        )

    def __init__(self,
                 path_to_dataset_folder,
                 shape=(224, 224),
                 augmentations=False):
        images_folders = [
            os.path.join(path_to_dataset_folder, p)
            for p in os.listdir(path_to_dataset_folder)
            if self.folder_filter(p)
        ]

        self.num_classes = len(images_folders)

        self.images_data = [
            {
                'img_path': os.path.join(folder_path, image_name),
                'class': i
            }
            for i, folder_path in enumerate(images_folders)
            for image_name in os.listdir(folder_path)
        ]

        self.shape = shape

        self.preprocessing = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(self.shape, interpolation=2),
                torchvision.transforms.ToTensor()
            ]
        )

        self.augmentations = torchvision.transforms.Compose(
            [
                torchvision.transforms.ColorJitter(),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.RandomAffine(
                    15, translate=None,
                    scale=None, shear=None,
                    resample=False, fillcolor=0
                ),
                torchvision.transforms.RandomPerspective(
                    distortion_scale=0.1, p=0.5, interpolation=Image.NEAREST
                ),
                torchvision.transforms.Resize(
                    self.shape,
                    interpolation=Image.NEAREST
                ),
                torchvision.transforms.RandomChoice(
                    [
                        torchvision.transforms.CenterCrop(self.shape[0] - k)
                        for k in range(0, int(
                        self.shape[0] * 0.05), 1)
                    ]
                ),
                torchvision.transforms.Resize(
                    self.shape,
                    interpolation=Image.NEAREST
                ),
                torchvision.transforms.RandomGrayscale(p=0.1),
                torchvision.transforms.ToTensor()
            ]
        ) if augmentations else None

    def __len__(self):
        return len(self.images_data)

    def apply_augmentations(self, img):
        if self.augmentations is not None:
            return (torch.clamp(self.augmentations(img), 0, 1) - 0.5) * 2
        return (self.preprocessing(img) - 0.5) * 2

    def __getitem__(self, idx):
        selected_item = self.images_data[idx]
        image = Image.open(selected_item['img_path'])

        return self.apply_augmentations(image), selected_item['class']

