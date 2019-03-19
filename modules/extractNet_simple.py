# extractNet_simple.py
# Contains Simple Autoencoder model (Encoder 3, Decoder 3)


import torch.nn as nn


class extractNet_simple(nn.Module):
	def __init__(self):
		super(extractNet_simple, self).__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(3, 16, 3, stride=2, padding=1),
			nn.ReLU(),
			nn.Conv2d(16, 32, 3, stride=2, padding=1),
			nn.ReLU(),
			nn.Conv2d(32, 64, 7)
		)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(64, 32, 7),
			nn.ReLU(),
			nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
			nn.ReLU(),
			nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
			nn.Sigmoid()
		)


	def forward(self, img):
		out = self.encoder(img)
		out = self.decoder(out)
		return out

