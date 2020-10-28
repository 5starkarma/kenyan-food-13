import os

from dataset import KenyanFood13Dataset
from augmentation import (
    image_resize, 
    image_preprocess, 
    common_transforms, 
    data_aug, 
    extra_data_aug
)
from settings import cfg

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import pandas as pd


def prediction(model, batch_input):

    model.to(cfg.device)
    model.eval()

    data = batch_input.to(cfg.device)

    bs, ncrops, c, h, w = data.size()
    output = model(data.view(-1, c, h, w))
    output = output.view(bs, ncrops, -1).mean(1) #   <----- max?
    prob = F.softmax(output, dim=1)
    pred_prob = prob.data.max(dim=1)[0]
    pred_index = prob.data.max(dim=1)[1]

    return pred_index.cpu().numpy(), pred_prob.cpu().numpy()


def average_multiple_predictions(models, inputs, targets):      
    pred_prob = prob.data.max(dim=1)[0]
    pred_index = prob.data.max(dim=1)[1]
    return pred_index.cpu().numpy(), pred_prob.cpu().numpy()


def get_results(model):  
    train_dataset = KenyanFood13Dataset(common_transforms(cfg.mean, cfg.std))
    test_dataset = KenyanFood13Dataset(transform=common_transforms(cfg.mean, cfg.std), train=False)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers)

    predictions = []
    
    for batch in test_loader:
        idx, prob = prediction(model, batch_input=batch)

        for target in idx:
            predictions.append(train_dataset.index_to_class(target))

    classes = pd.DataFrame(predictions, columns=['class'])
    result = test_dataset.label_df.join(classes)
    
    return result


def combine_and_get_results(models):  
      
    train_dataset = KenyanFood13Dataset(common_transforms(cfg.mean, cfg.std))
    test_dataset = KenyanFood13Testset(transform=common_transforms(cfg.mean, cfg.std))
    
    test_loader = DataLoader(test_dataset, 
                             batch_size=cfg.batch_size, 
                             shuffle=False, 
                             num_workers=cfg.num_workers)

    model_1, model_2, model_3, model_4, model_5 = models
    
    model_1.to(cfg.device)
    model_1.eval()
    model_2.to(cfg.device)
    model_2.eval()
    model_3.to(cfg.device)
    model_3.eval()
    model_4.to(cfg.device)
    model_4.eval()
    model_5.to(cfg.device)
    model_5.eval()

    predictions = []

    for data in test_loader:

        data = data.to(cfg.device)

        output_m1 = model_1(data)
        output_m2 = model_2(data)
        output_m3 = model_3(data)
        output_m4 = model_4(data)
        output_m5 = model_5(data)

        output = torch.sum(output_m1, output_m2, output_m3, output_m4, output_m5) / 5.

        outputs = F.softmax(output, 1)

        pred_prob = outputs.data.max(dim=1)[0]
        pred_index = outputs.data.max(dim=1)[1]

        pred_index.cpu().numpy()
        pred_prob.cpu().numpy()

        for target in pred_index:
            predictions.append(train_dataset.index_to_class(target))

    classes = pd.DataFrame(predictions, columns=['class'])
    result = test_dataset.label_df.join(classes)

    return result

