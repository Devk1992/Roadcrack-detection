from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from GaborConv2dLayer import GaborConv2dLayer

class GaborPretrainedModel():
    def __init__(self):
        super(GaborPretrainedModel, self).__init__()
        
        
    def VGGNET16():
        vgg16 = models.vgg16(pretrained=True)
        ## freeze the layers
        for param in vgg16.parameters():
            param.requires_grad = False

        # Modify the last layer
        number_features = vgg16.classifier[6].in_features
        features = list(vgg16.classifier.children())[:-1] # Remove last layer
        features.extend([torch.nn.Linear(number_features, 2)])
        vgg16.classifier = torch.nn.Sequential(*features)
        layers=list(vgg16.features.children())
        layers.extend([GaborConv2dLayer(512, 512, kernel_size=(3, 3), stride=1),
                      nn.ReLU(inplace=True),
                      nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)])
        vgg16.features= torch.nn.Sequential(*layers)
        return vgg16
    

    def RESNET18():
        resnet18 = models.resnet18(pretrained=True)
        for param in resnet18.parameters():
            param.requires_grad = False
         
        return resnet18

