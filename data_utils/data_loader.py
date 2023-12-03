import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split

class MyDataset(Dataset):
    def __init__(self, config, mode):
        if mode == 'train':
            data_path = config['train_path']
        elif mode == 'val':
            data_path = config['val_path']
        else:
            data_path = config['test_path']
        self.data = datasets.ImageFolder(
            root = data_path,
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        )                                                                                                                       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return image, label

class getDataloader():
    def __init__(self, config):
        self.config = config

        test_dataset = MyDataset(self.config, 'test')

        if config['val_path'] != 'None':
            val_path=config['val_path']
            self.train_dataset = MyDataset(self.config, 'train')
            self.val_dataset = MyDataset(self.config, 'val')
        else:
            train_val_dataset = MyDataset(self.train_path, self.config)
            dataset_size = len(train_val_dataset)
            val_size = int(0.1 * dataset_size)
            train_size = dataset_size - val_size
            self.train_dataset, self.val_dataset = random_split(train_val_dataset, [train_size, val_size])

    def get_train(self):
        return DataLoader(self.train_dataset, batch_size = self.config['batch_size'], shuffle = self.config['shuffle'])

    def get_val(self):
        return DataLoader(self.val_dataset, batch_size = self.config['batch_size'], shuffle = self.config['shuffle'])

    def get_test(self):
        return DataLoader(self.test_dataset, batch_size = self.config['batch_size'], shuffle = self.config['shuffle'])
