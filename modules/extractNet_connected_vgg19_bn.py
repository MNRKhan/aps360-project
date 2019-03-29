# extractNet_connected_vgg19_bn.py
# Contains Interconnected Autoencoder model (Encoder VGG19BN, Decoder VGG-mirror)


import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_blocks import *

encode_out = []
def hook(module, input, output):
	encode_out.append(output)

class extractNet_connected_vgg19_bn(nn.Module):
	def __init__(self, act_type = 'R'):
		super(extractNet_connected_vgg19_bn, self).__init__()

		vgg19_bn = torchvision.models.vgg.vgg19_bn(pretrained=True)

		# Maxpool output layers
		self.encoder_out_layers = [6,13,26,39,52]

		self.vgg = vgg19_bn

		# Freeze weights
		for param in self.vgg.features.parameters():
			param.requires_grad = False

		# Save intermediate output values
		for i in self.encoder_out_layers:
			self.vgg.features[i].register_forward_hook(hook)

		self.deconv1 = deconvBlock(512, 512, 3, stride=2, padding=1, output_padding=1, act_type = act_type)

		self.deconv2 = deconvBlock(512+512, 256, 3, stride=2, padding=1, output_padding=1, act_type = act_type)

		self.deconv3 = deconvBlock(256+256, 128, 3, stride=2, padding=1, output_padding=1, act_type = act_type)

		self.deconv4 = deconvBlock(128+128, 64, 3, stride=2, padding=1, output_padding=1, act_type = act_type)

		self.deconv5 = deconvBlock(64+64, 3, 3, stride=2, padding=1, output_padding=1, act_type = act_type)

		self.deconv6 = deconvBlock(3, 1, 3, stride=1, padding=1, with_activation = False, act_type = act_type)


	def forward(self, img):
		encode_out.clear()

		out = self.vgg.features(img)

		out = self.deconv1(encode_out[-1])

		out = torch.cat((out, encode_out[-2]), 1)
		out = self.deconv2(out)

		out = torch.cat((out, encode_out[-3]),1)
		out = self.deconv3(out)

		out = torch.cat((out, encode_out[-4]),1)
		out = self.deconv4(out)

		out = torch.cat((out, encode_out[-5]),1)
		out = self.deconv5(out)

		#out = torch.cat((out, img),1)
		out = self.deconv6(out)

		return out

