import timm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Perceptual_net(nn.Module):
    def __init__(self):
        super(Perceptual_net, self).__init__()
        self.net = timm.create_model('resnet101', pretrained = True, in_chans = 1).cuda().eval()

        self.layer1 = nn.Sequential(*(list(self.net.children())[:5]))
        self.layer2 = nn.Sequential(list(self.net.children())[5])
        self.layer3 = nn.Sequential(list(self.net.children())[6])
        self.layer4 = nn.Sequential(list(self.net.children())[7])
    def forward(self, img):
        layer1 = self.layer1(img)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        return [layer1, layer2, layer3, layer4]