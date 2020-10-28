import os
import time
import numpy as np

from dataset import KenyanFood13Dataset, KenyanFood13Subset
from models import save_model, load_model, pretrained_net
from settings import cfg
from augmentation import data_aug, common_transforms, get_training_augmentation

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF
from torchvision import transforms

import tensorboard as tb
from sklearn.model_selection import KFold


 
tb_writer = SummaryWriter(cfg.log_path)


def get_mean_std():
    """Gets mean and standard deviation of the dataset images."""     
    dataset = KenyanFood13Dataset(transform=image_preprocess())
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    
    mean = 0.0
    std = 0.0
    for images, _ in loader:
        batch_samples = images.size(0) # the last batch can have smaller size
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)

    return mean, std


def get_data():
    """Loads and splits data into train and test subsets. """  
    dataset = KenyanFood13Dataset()
    
    if cfg.data_augmentation:
        train_transforms = get_training_augmentation()
    else:
        train_transforms = common_transforms(cfg.mean, cfg.std)
    test_transforms = common_transforms(cfg.mean, cfg.std)

    validation_split = 0.2
    dataset_size = dataset.__len__()
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(cfg.seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_dataset = KenyanFood13Subset(dataset, train_indices, train_transforms)
    test_dataset = KenyanFood13Subset(dataset, val_indices, test_transforms)

    train_targets, train_class_counts = train_dataset.get_labels_and_class_counts()
    weights = 1. / torch.tensor(train_class_counts, dtype=torch.float)
    samples_weights = weights[train_targets]

    sampler = WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(samples_weights),
        replacement=True)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        sampler=sampler, 
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available())  

    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available())
    
    return train_loader, test_loader


def train(device, model, optimizer, train_loader, epoch_idx):  
    """Train loop for the model"""
    model.train()

    batch_loss = np.array([])
    batch_acc = np.array([])        
    for batch, (data, target) in enumerate(train_loader):

        target_index = target.clone()
        data, target = data['image'].to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        batch_loss = np.append(batch_loss, [loss.item()])
        prob = F.softmax(output, dim=1)
        pred = prob.data.max(dim=1)[1]  
        correct = pred.cpu().eq(target_index).sum()
        acc = float(correct) / float(len(data))
        batch_acc = np.append(batch_acc, [acc])
            
    epoch_loss = batch_loss.mean()
    epoch_acc = 100. * batch_acc.mean()
    print(f'Training   - loss: {epoch_loss:.4f}, accuracy: {epoch_acc:.2f}%')

    return epoch_loss, epoch_acc


def validate(model, device, test_loader):
    """Validation loop for the model"""
    model.eval()

    loss = 0.0
    correct = 0.0
    for data, target in test_loader:

        target_index = target.clone()
        data, target = data.to(device), target.to(device)
        bs, ncrops, c, h, w = data.size()

        output = model(data.view(-1, c, h, w))
        output = output.view(bs, ncrops, -1).mean(1) #   <----- max?
        
        loss += F.cross_entropy(output, target).item()
        prob = F.softmax(output, dim=1)
        pred = prob.data.max(dim=1)[1] 
        correct += pred.cpu().eq(target_index).sum()

    loss = loss / len(test_loader)  
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Validation - loss: {loss:.4f}, accuracy: {accuracy:.2f}%, {correct}/{len(test_loader.dataset)}')
    return loss, accuracy


def main(tb_writer):
    """Combines training, validation, and logging"""
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        batch_size_to_set = 10
        num_workers_to_set = 2

    print(f'Device: {device}')
    print(f'Folds: {cfg.splits}')
    print(f'Epochs: {cfg.epochs_count}')
    print(f'Batch size: {cfg.batch_size}')
    print(f'Data Augmentation: {cfg.data_augmentation}')
    print(f'Scheduler step size: {cfg.scheduler_step_size}')
    print(f'Scheduler gamma: {cfg.scheduler_gamma}')
    print(f'Learning rate: {cfg.init_learning_rate}')
    print(f'L2 weight decay: {cfg.weight_decay}')
    print(f'Model description: {cfg.description}')
    
    train_loader, test_loader = get_data()
    
    t_begin = time.time()   
    
    model = pretrained_net()  
    
    try:
        model = load_model(model)
    except:
        pass

    best_loss = torch.tensor(np.inf)

    epoch_train_loss = np.array([])
    epoch_test_loss = np.array([])
    epoch_train_acc = np.array([])
    epoch_test_acc = np.array([])
    
    model.to(device)

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.init_learning_rate,
        weight_decay=cfg.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,                                           
        step_size=cfg.scheduler_step_size, 
        gamma=cfg.scheduler_gamma)

    for epoch in range(cfg.epochs_count):
        print(f'\nEpoch: {epoch + 1}/{cfg.epochs_count}')  
          
        train_loss, train_acc = train(
            device=device,
            model=model, 
            optimizer=optimizer, 
            train_loader=train_loader, 
            epoch_idx=epoch)
        
        epoch_train_loss = np.append(epoch_train_loss, [train_loss])
        epoch_train_acc = np.append(epoch_train_acc, [train_acc])
        elapsed_time = time.time() - t_begin
        speed_epoch = elapsed_time / (epoch + 1)
        speed_batch = speed_epoch / len(train_loader)
        eta = speed_epoch * cfg.epochs_count - elapsed_time

        tb_writer.add_scalar('Loss/Train', train_loss, epoch)
        tb_writer.add_scalar('Accuracy/Train', train_acc, epoch)
        tb_writer.add_scalar('Time/elapsed_time', elapsed_time, epoch)
        tb_writer.add_scalar('Time/speed_epoch', speed_epoch, epoch)
        tb_writer.add_scalar('Time/speed_batch', speed_batch, epoch)
        tb_writer.add_scalar('Time/eta', eta, epoch)

        if epoch % cfg.test_interval == 0:
            current_loss, current_acc = validate(
                model, 
                device, 
                test_loader)

            epoch_test_loss = np.append(epoch_test_loss, [current_loss])
            epoch_test_acc = np.append(epoch_test_acc, [current_acc])

            if current_loss < best_loss:
                best_loss = current_loss
                save_model(model, device=device)
                print('----------Model Improved! Saved!----------')

            tb_writer.add_scalar('Loss/Validation', current_loss, epoch)
            tb_writer.add_scalar('Accuracy/Validation', current_acc, epoch)
            tb_writer.add_scalars('Loss/train-val', {'train': train_loss, 'validation': current_loss}, epoch)
            tb_writer.add_scalars('Accuracy/train-val', {'train': train_acc,'validation': current_acc}, epoch)

            if scheduler is not None:
                scheduler.step()   
                
            print(f'Time: {elapsed_time:.2f}s, {speed_epoch:.2f} s/epoch, {speed_batch:.2f} s/batch, Learning rate: {scheduler.get_last_lr()[0]}') 
    print(f'Total time: {time.time() - t_begin:.2f}, Best loss: {best_loss:.3f}')    
    return model


