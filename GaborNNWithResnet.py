from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from GaborConv2dLayer import GaborConv2dLayer

class GaborNNWithResnet(nn.Module):
    def __init__(self):
        super(GaborNNWithResnet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.gaborLayer= nn.Sequential(
                       GaborConv2dLayer(512, 512, kernel_size=(3, 3), stride=1),
                      nn.ReLU(inplace=True),
                      nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                       )
        self.classifier_layer = nn.Sequential(
            nn.Linear(512 , 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256 , 128),
            nn.Linear(128 , 2)
        )

    def forward(self, x):
        batch_size ,_,_,_ = x.shape #taking out batch_size from input image
        x = self.features(x)
        x = self.gaborLayer(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x,1).reshape(batch_size,-1) # then reshaping the batch_size
        x = self.classifier_layer(x)
        return x
    
    def features(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x

    def logitsss(self, features,model):
        x = self.resnet.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.resnet.last_linear(x)
        return x

    def forwardss(self, x,model):
        x = self.resnet.features(input)
        x = self.resnet.logits(x)
        return x

    def _forward_unimplemented(self, *inputs: Any):
        """
        code checkers makes implement this method,
        looks like error in PyTorch
        """
        raise NotImplementedError
