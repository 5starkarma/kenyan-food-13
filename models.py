import os

from settings import cfg

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Ensemble(nn.Module):
    def __init__(self, modelA, modelB, modelC, nb_classes=13):
        super(Ensemble, self).__init__()

        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        # self.modelD = modelD
        # self.modelE = modelE

        self.modelA.fc = nn.Identity()
        self.modelB.classifier = nn.Identity()
        self.modelC.fc = nn.Identity()
        # self.modelD.fc = nn.Identity()
        # self.modelE.fc = nn.Identity()
        
        self.classifier = nn.Linear(2048+2048+1024, cfg.num_classes)
        
    def forward(self, x):
        x1 = self.modelA(x.clone())
        x1 = x1.view(x1.size(0), -1)

        x2 = self.modelB(x.clone())
        x2 = x2.view(x2.size(0), -1)

        if self.modelC.training:
            x3, aux = self.modelC(x.clone())
        else:
            x3 = self.modelC(x.clone())
        x3 = x3.view(x3.size(0), -1)

        # x4 = self.modelD(x.clone())
        # x4 = x4.view(x4.size(0), -1)

        # x5 = self.modelE(x.clone())
        # x5 = x5.view(x5.size(0), -1)

        x = torch.cat((x1, x2, x3), dim=1)

        x = self.classifier(F.relu(x))

        return x    


def trainable_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


def pretrained_net():   
    # modelA = models.densenet121(pretrained=True, progress=True)
    # modelB = models.densenet161(pretrained=True, progress=True)
    # modelC = models.densenet169(pretrained=True, progress=True)
    # modelD = models.densenet121(pretrained=True, progress=True)
    # modelE = models.resnet34(pretrained=True, progress=True)

    # for name, param in modelD.named_parameters():
    #     if param.requires_grad:
    #         print(name)


    modelA = models.resnext50_32x4d(pretrained=True)
    modelB = models.densenet121(pretrained=True)
    modelC = models.inception_v3(pretrained=True)
    # modelD = models.resnet18(pretrained=True)
    # modelE = models.wide_resnet50_2(pretrained=True)
    # modelC = models.resnext50_32x4d(pretrained=True, progress=True)
    # modelD = models.resnext101_32x8d(pretrained=True, progress=True)
    # modelE = models.resnet34(pretrained=True, progress=True)

    # for name, param in modelD.named_parameters():
    #     print(name)

    for param in modelA.parameters():
        param.requires_grad_(False)

    for param in modelB.parameters():
        param.requires_grad_(False)

    for param in modelC.parameters():
        param.requires_grad_(False)

    # for param in modelD.parameters():
    #     param.requires_grad_(False)

    # for param in modelE.parameters():
    #     param.requires_grad_(False)

    layers = [
        modelA.layer4,
        modelB.features.denseblock4.denselayer16,
    ]
    
    for layer in layers:
        for param in layer.parameters():
            param.requires_grad = True

    model = Ensemble(modelA, modelB, modelC)

#     model_path = os.path.join(tc.model_dir, tc.model_name)
#     pretrained_model_path = os.path.join(tc.root_dir,
#                                          tc.model_dir, 
#                                          tc.model_name + '_pretrained.pt')

#     try:
#         model = torch.load(pretrained_model_path)

#     except FileNotFoundError:
#         os.environ['TORCH_HOME'] = tc.root_dir
#         model = models.densenet121(pretrained=True, progress=True)
#         torch.save(model, pretrained_model_path)
#         model = torch.load(pretrained_model_path)
    
    return model


def save_model(model, device):
    model_dir = os.path.join(cfg.root_dir, cfg.model_dir)
    model_path = os.path.join(model_dir, cfg.description)
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if device == 'cuda':
        model.to('cpu')

    torch.save(model.state_dict(), model_path + '.pt' )

    if device == 'cuda':
        model.to('cuda')
    return


def load_model(model):  
    model_path = os.path.join(cfg.root_dir, cfg.model_dir, cfg.description)
    model.load_state_dict(torch.load(model_path + '.pt')) 
    return model

def load_multiple_models(model):

    model_1 = load_model(model=model, fold=0)
    model_2 = load_model(model=model, fold=1)
    model_3 = load_model(model=model, fold=2)
    model_4 = load_model(model=model, fold=3)
    model_5 = load_model(model=model, fold=4)

    models = [model_1, model_2, model_3, model_4, model_5]

    return models

