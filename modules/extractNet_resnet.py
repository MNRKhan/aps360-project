# extractNet_resnet.py
# Contains Interconnected Autoencoder model (Encoder ResNet, Decoder Resnet-mirror)


import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

encode_out_r = []


def hook_r(module, input, output):
    encode_out_r.append(output)


class extractNet_resnet(nn.Module):
    def __init__(self):
        super(extractNet_resnet, self).__init__()

        resnet = torchvision.models.resnet152(pretrained=True)

        # Maxpool output layers
        self.encoder_out_layers = [resnet.maxpool,
                                   resnet.layer1[0].downsample[-1],
                                   resnet.layer2[0].downsample[-1],
                                   resnet.layer3[0].downsample[-1],
                                   resnet.layer4[0].downsample[-1]]

        self.res = resnet

        # Freeze weights
        for param in self.res.parameters():
            param.requires_grad = False

        # Save intermediate output values
        for layer in self.encoder_out_layers:
            layer.register_forward_hook(hook_r)

        self.deconv1 = nn.ConvTranspose2d(2048, 1024, 3, stride=2, padding=1, output_padding=1)

        self.deconv2 = nn.ConvTranspose2d(1024 + 1024, 512, 3, stride=2, padding=1, output_padding=1)

        self.deconv3 = nn.ConvTranspose2d(512 + 512, 256, 3, stride=2, padding=1, output_padding=1)

        self.deconv4 = nn.ConvTranspose2d(256 + 256, 64, 3, stride=2, padding=1, output_padding=1)

        self.deconv5 = nn.ConvTranspose2d(64 + 64, 3, 3, stride=2, padding=1, output_padding=1)

        self.deconv6 = nn.ConvTranspose2d(3 + 3, 1, 3, stride=1, padding=1)

    def forward(self, img):
        encode_out_r.clear()

        out_res = self.res(img)

        out = F.relu(self.deconv1(encode_out_r[-1]))

        out = torch.cat((out, encode_out_r[-2]), 1)
        out = F.relu(self.deconv2(out))

        out = torch.cat((out, encode_out_r[-3]), 1)
        out = F.relu(self.deconv3(out))

        out = torch.cat((out, encode_out_r[-4]), 1)
        out = F.relu(self.deconv4(out))

        out = torch.cat((out, encode_out_r[-5]), 1)
        out = F.relu(self.deconv5(out))

        out = torch.cat((out, img), 1)
        out = self.deconv6(out)

        return out
