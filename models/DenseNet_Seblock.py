import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding = 1, dilation = 1, is_seblock = False, is_last=False):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        self.is_last = is_last
        self.is_seblock = is_seblock

    def forward(self, x):
        out = self.conv2d(x)
        if self.is_seblock is True and self.is_last is True:
            out = F.relu(out, inplace=True)
        elif self.is_last is True:
            out = F.tanh(out)
        else:
            out = F.leaky_relu(out, inplace=True)
        return out

# Dense convolution unit
class DenseConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseConv2d, self).__init__()
        self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out


# Dense Block unit
class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(DenseBlock, self).__init__()
        out_channels_def = 16
        denseblock = []
        denseblock += [DenseConv2d(in_channels, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def*2, out_channels_def, kernel_size, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out

class SEBlock(nn.Module):
    def __init__(self):
        super(SEBlock, self).__init__()
        kernel_size = 3
        stride = 1

        self.conv1 = ConvLayer(64, 8, kernel_size, stride)
        self.conv2 = ConvLayer(8, 64, kernel_size, stride, is_seblock = True, is_last=True)
        
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, img):
        f_1 = self.conv1(img)
        f_2 = self.conv2(f_1)

        vector_f = torch.mean(f_2, [2,3], True)
        vector_f = self.softmax(vector_f)
        return vector_f


#  DenseFuse network
class DenseNet(nn.Module):
    def __init__(self, input_nc=6, output_nc=1):
        super(DenseNet, self).__init__()
        denseblock = DenseBlock
        nb_filter = [16, 64, 32, 16]
        kernel_size = 3
        stride = 1

        # encoder
        self.conv1 = ConvLayer(input_nc, nb_filter[0], kernel_size, stride)
        self.DB1 = denseblock(nb_filter[0], kernel_size, stride)

        self.seblock = SEBlock()

        # decoder
        self.conv_de = ConvLayer(nb_filter[1], nb_filter[1], kernel_size, stride)
        self.conv2 = ConvLayer(nb_filter[1], nb_filter[1], kernel_size, stride)
        self.conv3 = ConvLayer(nb_filter[1], nb_filter[2], kernel_size, stride)
        self.conv4 = ConvLayer(nb_filter[2], nb_filter[3], kernel_size, stride)
        self.conv5 = ConvLayer(nb_filter[3], output_nc, kernel_size, stride, is_last=True)

    def forward(self, ir, vi):
        _input = torch.cat([ir, vi], axis = 1)
        f_en = self.conv1(_input)
        f_en = self.DB1(f_en)
        
        vector = self.seblock(f_en)

        _en = f_en*vector

        _de = self.conv_de(_en)
        _de = self.conv2(_de)
        _de = self.conv3(_de)
        _de = self.conv4(_de)
        output = self.conv5(_de)

        return output
