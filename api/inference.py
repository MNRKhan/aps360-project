import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from utils import *
from model_blocks import *
import cv2
from keras.preprocessing.image import array_to_img
from PIL import Image

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


def extract(in_path: str):

    with torch.no_grad():

        img = get_img(in_path)

        print("##############INNNNNN################################")
        print(img.shape)

        path = "/floyd/input/weights/resnet"

        net = extractNet_resnet_prelu()
        net = load_state_from_dc(net, path)
        net = net.eval()

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        input = transform(img).unsqueeze(0)

        out = net(input)

        out = out.squeeze(0).squeeze(0).detach().numpy()

        return get_masked(img, out)


# Both are numpy arrays

def get_masked(original, mask, save=False):

    mask = torch.sigmoid(torch.Tensor(mask))
    mask = thresholdProbMask(mask.squeeze(0).squeeze(0).detach().numpy())

    mask = cv2.blur(mask, (27, 27))
    mask = thresholdProbMask(mask, threshold=0.5)

    mask = np.array(mask, dtype=np.uint8)
    mask = denoise(mask, kernel_size=39)

    mask = np.expand_dims(mask, axis=-1)

    final = mask * original
    final = np.dstack((final, 255*mask))
    final = final.astype(int)

    print("##############################################")
    print(final.shape)

    img = Image.fromarray(final.astype('uint8'))

    return img


def thresholdProbMask(prob_mask, threshold=0.5):

    prob_mask[prob_mask > threshold] = 1
    prob_mask[prob_mask <= threshold] = 0

    return prob_mask