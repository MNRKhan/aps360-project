# data_loader.py
# Contains functions for using cleaned dataset

import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms

# PREREQUISITE: Must have ran 'dataset_downloader_script' prior to this

# Custom class for cleaned dataset

class ImageMaskDataset(Dataset):

  def __init__(self, data_dir='/content/data', transform=None):

    self.data_dir = data_dir
    self.transform = transform


  def __len__(self):

    return len(
      [name for name in os.listdir(self.data_dir + '/images') if os.path.isfile(self.data_dir + '/images/' + name)])


  def __getitem__(self, idx):

    ids = np.atleast_1d(idx.detach().numpy())

    item = []

    for id in ids:

      img = plt.imread(self.data_dir + "/images/" + str(id) + ".jpg")

      if self.transform:
        img = self.transform(img)

      mask = torch.Tensor(np.load(self.data_dir + "/masks/" + str(id) + ".npy")).unsqueeze(0)

      item.append((img, mask))

    if len(item) == 1:
      return item[0]

    return item


# Example usage

def main():

  transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  dataset = ImageMaskDataset(data_dir="/content/data", transform=transform)
  size = len(dataset)

  train_size = int(0.6 * size)
  valid_size = int(0.2 * size)
  test_size = size - train_size - valid_size

  print(train_size, valid_size, test_size)

  train, valid, test = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

  train = train[:10]

  for i, (img, mask) in enumerate(train):

    if i > 10:
      break

    print(img.shape)
    print(mask.shape)