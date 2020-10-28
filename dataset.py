import os

from settings import cfg

import pandas as pd
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, SubsetRandomSampler
import torch



class KenyanFood13Dataset(Dataset):
    """Kenyan food dataset."""
    def __init__(self, transform=None, train=True):
        self.train = train
        if self.train:
            self.label_df = pd.read_csv(cfg.train_csv_path)
        else:
            self.label_df = pd.read_csv(cfg.test_csv_path)
        self.transform = transform
        self.classes = list(self.label_df['class'].unique())

    def __getitem__(self, idx):
        """Return (image, target) after resize and preprocessing."""
        image = os.path.join(
            cfg.root_dir, 
            cfg.trial_img_dir, 
            str(self.label_df.iloc[idx, 0]) + '.jpg')

        if os.path.isfile(image):
            pass
        else:
            image = os.path.join(
                cfg.root_dir, 
                cfg.img_dir, 
                str(self.label_df.iloc[idx, 0]) + '.jpg')

        image = Image.open(image)

        if self.transform:
            image = self.transform(image)

        if self.train:
            y = self.class_to_index(self.label_df.iloc[idx, 1])
            return image, y
        else:
            return image
    
    def class_to_index(self, class_name):
        """Returns the index of a given class."""
        return self.classes.index(class_name)
    
    def index_to_class(self, class_index):
        """Returns the class of a given index."""
        return self.classes[class_index] 

    def display_class_images(self):
        imgs = []
        for i in self.classes:
            print(i)
    
    def get_class_count(self):
        """Return a list of label occurences"""
        cls_count = self.label_df.groupby('class')['id'].nunique().to_list()
        floated_cls_count = [float(x) for x in cls_count]
        return floated_cls_count
    
    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.label_df)



class KenyanFood13Subset(Dataset):
    """Subset of a dataset at specified indices."""
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.labels = pd.read_csv(cfg.train_csv_path)
        self.transform = transform

    def __getitem__(self, idx):
        image, target = self.dataset[self.indices[idx]]

        if self.transform:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.indices)

    def map_labels(self, label_list):
        label_map = {
            'bhaji': 0,
            'chapati': 1, 
            'githeri': 2, 
            'kachumbari': 3, 
            'kukuchoma': 4, 
            'mandazi': 5,
            'masalachips': 6, 
            'matoke': 7, 
            'mukimo': 8, 
            'nyamachoma': 9, 
            'pilau': 10, 
            'sukumawiki': 11, 
            'ugali': 12
        }
        return np.array(list(map(label_map.get, label_list)))


    def get_labels_and_class_counts(self):
        labels = self.labels[self.labels.index.isin(self.indices)]
        labels_list = self.map_labels(labels['class'])
        class_counts = np.array(labels.groupby('class')['id'].nunique().to_list())
        return labels_list, class_counts
