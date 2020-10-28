from autoaugment import ImageNetPolicy
import albumentations as A
import albumentations.pytorch as AP

from dataset import KenyanFood13Dataset, KenyanFood13Subset
from settings import cfg

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np


def get_training_augmentation():
    train_transform = [
        A.Resize(cfg.resize_size, cfg.resize_size),
        A.CenterCrop(cfg.input_size, cfg.input_size),
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomBrightnessContrast(),            
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
        A.Normalize(mean=cfg.mean, std=cfg.std),
        AP.ToTensor(),
    ]
    transforms = A.Compose(train_transform)
    return lambda img:transforms(image=np.array(img))


def image_resize():
    """
    Transforms for resizing, cropping.
    """
    resize_transforms = transforms.Compose([transforms.Resize(cfg.resize_size),
                                            transforms.CenterCrop(cfg.input_size),
                                           ])
    return resize_transforms

def image_preprocess():
    """
    Transforms for resizing, cropping, then converting to Tensor.
    """
    preprocess_transforms = transforms.Compose([transforms.Resize(cfg.resize_size),
                                                transforms.CenterCrop(cfg.input_size),
                                                transforms.ToTensor()
                                               ])
    return preprocess_transforms

def common_transforms(mean, std):
    """
    Transforms which are common to both the train and test set.
    """ 
    common_transforms = transforms.Compose([
        transforms.Resize(cfg.resize_size),
        transforms.FiveCrop(cfg.input_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda tensors: torch.stack([transforms.Normalize(mean, std)(t) for t in tensors]))
        ])
    return common_transforms

def data_aug(mean, std):
    """
    Data augmentation transforms.
    """
    data_aug_transforms = A.Compose([
        # transforms.RandomResizedCrop(cfg.input_size),
        A.Resize(cfg.input_size, cfg.input_size),
        A.CenterCrop(cfg.input_size, cfg.input_size),
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomBrightnessContrast(),            
        ], p=0.3),
        A.HueSaturationValue(p=0.3),

        # A.RandomResizedCrop(cfg.input_size, cfg.input_size),
        #   A.CenterCrop(cfg.input_size, cf),
        # APy.transforms.ToTensor(),
        A.Normalize(mean, std)
    ])
    return data_aug_transforms


def extra_data_aug(mean, std):
    """
    Extra data augmentation transforms.
    """
    data_aug_transforms = transforms.Compose([transforms.Resize(cfg.resize_size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.FiveCrop(cfg.input_size), # this is a list of PIL Images
                                              transforms.ColorJitter(),
                                              transforms.Lambda(lambda crops: torch.stack(
                                                  [transforms.ToTensor()(crop) for crop in crops])),
                                              transforms.Lambda(lambda tensors: torch.stack(
                                                  [transforms.RandomErasing()(t) for t in tensors])),
                                              transforms.Lambda(lambda tensors: torch.stack(
                                                  [transforms.Normalize(mean, std)(t) for t in tensors])),
                                             ])
    return data_aug_transforms