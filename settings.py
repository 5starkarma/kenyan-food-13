import os

from dataclasses import dataclass

import torch


@dataclass
class Configurations:
    '''Configurations for the system'''
    seed: int = 11
    cudnn_benchmark_enabled: bool = True  # enable for sake of performance
    cudnn_deterministic: bool = True  # (reproducible training)

    device: str = 'cuda'
    num_workers: int = 8
    log_interval: int = 5  
    test_interval: int = 1  

    resize_size: int = 512
    input_size: int = 512
    num_classes: int = 13
    batch_size: int = 16
    epochs_count: int = 60
    splits: int = 5

    init_learning_rate: float = 0.0001
    weight_decay: float = 0.001
    scheduler_step_size: int = 15
    scheduler_gamma: float = 0.1

    mean = [0.5718, 0.4658, 0.3593]
    std = [0.2467, 0.2549, 0.2578]

    weights: bool = True
    data_augmentation: bool = True

    root_dir: str = '/content/drive/My Drive/Colab/001-classification/kenyan_food_13'

    model_name: str = 'resnext_densenet_inception_ensemble_segmented_wrs_weights'
    model_dir: str = 'output/models/'
    model_path: str = os.path.join(root_dir, model_dir) 

    description: str = model_name 
    # + '_' + str(resize_size) + '_bs-' + str(batch_size) + '_lr-001'  + '_L2-001'  + '_sch_step-12'# + str(scheduler_step_size) + str(init_learning_rate)[2:]+ str(weight_decay)[2:]

    submission_dir: str = 'output/submissions/'
    submission_csv: str = description + '_submission.csv'
    submission_path: str = os.path.join(root_dir, submission_dir, submission_csv) 

    log_dir: str = 'output/logs/'
    log_path: str = os.path.join(root_dir, log_dir, description) 

    train_csv_file: str = 'labels/train.csv'
    train_csv_path: str = os.path.join(root_dir, train_csv_file) 

    test_csv_file: str = 'labels/test.csv'
    test_csv_path: str = os.path.join(root_dir, test_csv_file) 

    img_dir: str = 'images/images/'
    trial_img_dir: str = 'images/output/'
    img_path: str = os.path.join(root_dir, img_dir) 


cfg = Configurations()

def setup_system():
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.backends.cudnn_benchmark_enabled = cfg.cudnn_benchmark_enabled
        torch.backends.cudnn.deterministic = cfg.cudnn_deterministic