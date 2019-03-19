# extractNet_connected.py
# Contains Interconnected Autoencoder model (Encoder 3, Decoder 3)


import torch
import torch.nn as nn
import torch.nn.functional as F


class extractNet_connected(nn.Module):
	def __init__(self):
		super(extractNet_connected, self).__init__()

		self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
		self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
		self.conv3 = nn.Conv2d(32, 64, 7)

		self.deconv1 = nn.ConvTranspose2d(64, 32, 7)
		self.deconv2 = nn.ConvTranspose2d(32+32, 16, 3, stride=2, padding=1, output_padding=1)
		self.deconv3 = nn.ConvTranspose2d(16+16, 1, 3, stride=2, padding=1, output_padding=1)


	def forward(self, img):
		enc_out1 = F.relu(self.conv1(img))
		enc_out2= F.relu(self.conv2(enc_out1))
		enc_out3 = F.relu(self.conv3(enc_out2))

		out = F.relu(self.deconv1(enc_out3))
		out = torch.cat((out, enc_out2),1)
		out = F.relu(self.deconv2(out))

		out = torch.cat((out, enc_out1),1)
		out = self.deconv3(out)

		out = torch.sigmoid(out)
		return out

