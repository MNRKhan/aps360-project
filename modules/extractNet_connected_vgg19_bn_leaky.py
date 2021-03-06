# extractNet_connected_vgg19_bn_leaky.py
# Contains Interconnected Autoencoder model (Encoder VGG19BN, Decoder VGG-mirror)


import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F


encode_out = []
def hook(module, input, output):
	encode_out.append(output)

class extractNet_connected_vgg19_bn_leaky(nn.Module):
	def __init__(self, alpha = 0.1):
		super(extractNet_connected_vgg19_bn_leaky, self).__init__()

		self.alpha = alpha

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

		self.deconv1 = nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1)

		self.deconv2 = nn.ConvTranspose2d(512+512, 256, 3, stride=2, padding=1, output_padding=1)

		self.deconv3 = nn.ConvTranspose2d(256+256, 128, 3, stride=2, padding=1, output_padding=1)

		self.deconv4 = nn.ConvTranspose2d(128+128, 64, 3, stride=2, padding=1, output_padding=1)

		self.deconv5 = nn.ConvTranspose2d(64+64, 3, 3, stride=2, padding=1, output_padding=1)

		self.deconv6 = nn.ConvTranspose2d(3+3, 1, 3, stride=1, padding=1)


	def forward(self, img):
		encode_out.clear()

		out = self.vgg.features(img)

		out = F.leaky_relu(self.deconv1(encode_out[-1]), self.alpha)

		out = torch.cat((out, encode_out[-2]), 1)
		out = F.leaky_relu(self.deconv2(out), self.alpha)

		out = torch.cat((out, encode_out[-3]),1)
		out = F.leaky_relu(self.deconv3(out), self.alpha)

		out = torch.cat((out, encode_out[-4]),1)
		out = F.leaky_relu(self.deconv4(out), self.alpha)

		out = torch.cat((out, encode_out[-5]),1)
		out = F.leaky_relu(self.deconv5(out), self.alpha)

		out = torch.cat((out, img),1)
		out = self.deconv6(out)

		return out

