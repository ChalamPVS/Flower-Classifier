import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models, datasets, transforms

class Net(nn.Module):
    def __init__(self, name):
        super(Net, self).__init__()
        model = self.feature_extractor(name)
        
        for params in model.parameters():
            params.requires_grad = False

        self.basemodel = model
        self.feedforward1 = nn.Linear(1000, 512)
        self.dropout1 = nn.Dropout(0.3)
        self.feedforward2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.feedforward3 = nn.Linear(128, 102)
    
    def forward(self, x):
        x = self.basemodel(x)
        x = F.relu(self.dropout1(self.feedforward1(x)))
        x = F.relu(self.dropout2(self.feedforward2(x)))
        x = self.feedforward3(x)
        return x

    def feature_extractor(self, name):
        if name == 'vgg16':
            model = models.vgg16(pretrained=True)
        elif name == 'resnet18':
            model = models.resnet18(pretrained=True)
        elif name == 'alexnet':
            model = models.alexnet(pretrained=True)
        elif name == 'squeezenet1_0':
            model = models.squeezenet1_0(pretrained=True)
        elif name == 'densenet161':
            model = models.densenet161(pretrained=True)
        else:
            raise ValueError('Should be one of vgg16, resnet18, alexnet, squeezenet1_0, densenet161')
        return model
        