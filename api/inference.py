import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from utils import *

encode_out = []
def hook(module, input, output):
	encode_out.append(output)

class extractNet_connected_vgg19(nn.Module):
	def __init__(self):
		super(extractNet_connected_vgg19, self).__init__()

		vgg19 = torchvision.models.vgg.vgg19(pretrained=True)

		# Maxpool output layers
		self.encoder_out_layers = [4,9,18,27,36]

		self.vgg = vgg19

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

		out = F.relu(self.deconv1(encode_out[-1]))

		out = torch.cat((out, encode_out[-2]), 1)
		out = F.relu(self.deconv2(out))

		out = torch.cat((out, encode_out[-3]),1)
		out = F.relu(self.deconv3(out))

		out = torch.cat((out, encode_out[-4]),1)
		out = F.relu(self.deconv4(out))

		out = torch.cat((out, encode_out[-5]),1)
		out = F.relu(self.deconv5(out))

		out = torch.cat((out, img),1)
		out = self.deconv6(out)

		return out

def extract(in_path, out_path):

	img = get_img(in_path)

	path = "layers_epoch_300"

	net = extractNet_connected_vgg19()
	net = load_state_from_dc(net, path)

	transform = transforms.Compose([transforms.ToTensor()])

	input = transform(img).unsqueeze(0)

	out = net(input)

	out = out.squeeze(0).squeeze(0).detach().numpy()

	get_masked(img, out, out_path)

# Both are numpy arrays

def get_masked(original, mask, out_path):

	mask = torch.sigmoid(torch.Tensor(mask))
	mask = thresholdProbMask(mask.squeeze(0).squeeze(0).detach().numpy())
	mask = np.expand_dims(mask, axis=-1)

	final = mask * original
	final = final.astype(int)

	save_img(out_path, final)


def thresholdProbMask(prob_mask, threshold=0.5):

	prob_mask[prob_mask > threshold] = 1
	prob_mask[prob_mask <= threshold] = 0

	return prob_mask