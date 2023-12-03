from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VGG_16(nn.Module):
    def __init__(self, config):
        super(VGG_16, self).__init__()
        self.num_classes = config['num_classes']
        self.cnn = models.vgg16(pretrained=config['pretrained'])
        if config['freeze']:
            for param in self.cnn.parameters():
                param.requires_grad=False
        self.cnn.classifier[-1] = nn.Linear(self.cnn.classifier[-1].in_features, self.num_classes)
        
    def forward(self, x):
        x = self.cnn(x)
        x = F.softmax(x, dim=-1)
        return x
    
class VGG_19(nn.Module):
    def __init__(self, config):
        super(VGG_19, self).__init__()
        self.num_classes = config['num_classes']
        self.cnn = models.vgg19(pretrained=config['pretrained'])
        if config['freeze']:
            for param in self.cnn.parameters():
                param.requires_grad=False
        self.cnn.classifier[-1] = nn.Linear(self.cnn.classifier[-1].in_features, self.num_classes)
        
    def forward(self, x):
        x = self.cnn(x)
        x = F.softmax(x, dim=-1)
        return x
    
class ResNet_18(nn.Module):
    def __init__(self, config):
        super(ResNet_18, self).__init__()
        self.num_classes = config['num_classes']
        self.cnn = models.resnet18(pretrained=config['pretrained'])
        if config['freeze']:
            for param in self.cnn.parameters():
                param.requires_grad=False
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, self.num_classes)
        
    def forward(self, x):
        x = self.cnn(x)
        x = F.softmax(x, dim=-1)
        return x

class ResNet_50(nn.Module):
    def __init__(self, config):
        super(ResNet_50, self).__init__()
        self.num_classes = config['num_classes']
        self.cnn = models.resnet50(pretrained=config['pretrained'])
        if config['freeze']:
            for param in self.cnn.parameters():
                param.requires_grad=False
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, self.num_classes)
        
    def forward(self, x):
        x = self.cnn(x)
        x = F.softmax(x, dim=-1)
        return x

class CNN_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config['model'] == 'vgg16':
            self.cnn = VGG_16(config)
        elif config['model'] == 'vgg19':
            self.cnn = VGG_19(config)
        elif config['model'] == 'resnet18':
            self.cnn = ResNet_18(config)
        else:
            self.cnn = ResNet_50(config)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, images, labels=None):
        if labels is not None:
            logits = self.cnn(images)
            loss = self.loss_fn(logits, labels)
            return logits, loss
        else:
            logits = self.cnn(images)
            return logits
