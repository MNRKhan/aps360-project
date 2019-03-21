# extractNet_connected_v2.py
# Contains Interconnected Autoencoder model (Encoder 4, Decoder 4)


import torch
import torch.nn as nn
import torch.nn.functional as F


class extractNet_connected_v2(nn.Module):
	def __init__(self):
		super(extractNet_connected_v2, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
		self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
		self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
		self.conv4 = nn.Conv2d(64, 128, 7)

		self.deconv1 = nn.ConvTranspose2d(128, 64, 7)
		self.deconv2 = nn.ConvTranspose2d(64+64, 32, 3, stride=2, padding=1, output_padding=1)
		self.deconv3 = nn.ConvTranspose2d(32+32, 16, 3, stride=2, padding=1, output_padding=1)
		self.deconv4 = nn.ConvTranspose2d(16+16, 1, 3, stride=2, padding=1, output_padding=1)


	def forward(self, img):
		enc_out1 = F.relu(self.conv1(img))
		enc_out2 = F.relu(self.conv2(enc_out1))
		enc_out3 = F.relu(self.conv3(enc_out2))
		enc_out4 = F.relu(self.conv4(enc_out3))

		out = F.relu(self.deconv1(enc_out4))

		out = torch.cat((out,enc_out3),1)
		out = F.relu(self.deconv2(out))

		out = torch.cat((out,enc_out2),1)
		out = F.relu(self.deconv3(out))

		out = torch.cat((out,enc_out1),1)
		out = self.deconv4(out)
		
		return out

