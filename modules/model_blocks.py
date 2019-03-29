#helper functions for model building
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

class deconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=2,
                 padding=1, output_padding=0, with_bn = False,
                 with_activation=True, act_type='R'):
        super().__init__()

        self.deconv_layer = nn.ConvTranspose2d(in_channels,
                                               out_channels,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               padding=padding,
                                               output_padding=output_padding)
        self.deconv_bn_layer = nn.BatchNorm2d(out_channels)

        if (act_type == 'R'):
            self.deconv_activation = nn.ReLU()
        else:
            self.deconv_activation = nn.PReLU()

        self.bn_flag = with_bn
        self.act_flag = with_activation

    def forward(self, input):

        out = self.deconv_layer(input)

        if self.bn_flag:
            out = self.deconv_bn_layer(out)

        if self.act_flag:
            out = self.deconv_activation(out)
        return out
