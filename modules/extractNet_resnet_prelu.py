
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_blocks import *

encode_out_r = []


def hook_r(module, input, output):
    encode_out_r.append(output)



class extractNet_resnet_prelu(nn.Module):
    def __init__(self, act_type = 'R', r_size = 50):
        super(extractNet_resnet_prelu, self).__init__()

        if r_size == 152:
            resnet = torchvision.models.resnet152(pretrained=True)
        elif r_size == 101:
            resnet = torchvision.models.resnet101(pretrained=True)
        else:
            resnet = torchvision.models.resnet50(pretrained=True)

        # Maxpool output layers
        self.encoder_out_layers = [resnet.conv1,
                                   resnet.maxpool,
                                   resnet.layer1[0].downsample[-1],
                                   resnet.layer2[0].downsample[-1],
                                   resnet.layer3[0].downsample[-1],
                                   resnet.layer4[-1].relu]

        self.res = nn.Sequential(*list(resnet.children())[:-2])

        # Freeze weights
        for param in self.res.parameters():
            param.requires_grad = False

        # Save intermediate output values
        for layer in self.encoder_out_layers:
            layer.register_forward_hook(hook_r)

        self.deconv1 = deconvBlock(2048, 1024, 3, stride=2, padding=1, output_padding=1, act_type = act_type)

        self.deconv2 = deconvBlock(1024 + 1024, 512, 3, stride=2, padding=1, output_padding=1, act_type = act_type)

        self.deconv3 = deconvBlock(512 + 512, 256, 3, stride=2, padding=1, output_padding=1, act_type = act_type)

        self.deconv4 = deconvBlock(256 + 256, 64, 3, stride=1, padding=1, act_type = act_type)

        self.deconv5 = deconvBlock(64 + 64, 64, 3, stride=2, padding=1, output_padding=1, act_type = act_type)

        self.deconv6 = deconvBlock(64 + 64, 3, 3, stride=2, padding=1, output_padding=1, act_type = act_type)

        self.deconv7 = deconvBlock(3, 1, 3, stride=1, padding=1,
                                   with_activation=False, act_type = act_type)

    def forward(self, img):
        global encode_out_r
        encode_out_r = []

        out_res = self.res(img)

        out = self.deconv1(encode_out_r[-1])

        # print(out.shape)

        out = torch.cat((out, encode_out_r[-4]), 1)
        out = self.deconv2(out)

        # print(out.shape)

        out = torch.cat((out, encode_out_r[-5]), 1)
        out = self.deconv3(out)

        # print(out.shape)

        out = torch.cat((out, encode_out_r[-6]), 1)
        out = self.deconv4(out)

        # print(out.shape)

        out = torch.cat((out, encode_out_r[-7]), 1)
        out = self.deconv5(out)

        # print(out.shape)

        out = torch.cat((out, encode_out_r[-8]), 1)
        out = self.deconv6(out)

        # print(out.shape)

        #out = torch.cat((out, img), 1)
        out = self.deconv7(out)

        return out