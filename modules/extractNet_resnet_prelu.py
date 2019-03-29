
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

encode_out_r = []


def hook_r(module, input, output):
    encode_out_r.append(output)


class deconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=2,
                 padding=1, output_padding=0, with_bn = True,
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


class extractNet_resnet_prelu(nn.Module):
    def __init__(self):
        super(extractNet_resnet_prelu, self).__init__()

        resnet = torchvision.models.resnet152(pretrained=True)

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

        self.deconv1 = deconvBlock(2048, 1024, 3, stride=2, padding=1, output_padding=1)

        self.deconv2 = deconvBlock(1024 + 1024, 512, 3, stride=2, padding=1, output_padding=1)

        self.deconv3 = deconvBlock(512 + 512, 256, 3, stride=2, padding=1, output_padding=1)

        self.deconv4 = deconvBlock(256 + 256, 64, 3, stride=1, padding=1)

        self.deconv5 = deconvBlock(64 + 64, 64, 3, stride=2, padding=1, output_padding=1)

        self.deconv6 = deconvBlock(64 + 64, 3, 3, stride=2, padding=1, output_padding=1)

        self.deconv7 = deconvBlock(3 + 3, 1, 3, stride=1, padding=1,
                                   with_activation=False)

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

        out = torch.cat((out, img), 1)
        out = self.deconv7(out)

        return out
