# data_loader.py
# Contains functions for using cleaned dataset

import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# PREREQUISITE: Must have ran 'dataset_downloader_script' prior to this

# Custom class for cleaned dataset

class ImageMaskDataset(Dataset):

  def __init__(self, data_dir, transform=None):

    self.data_dir = data_dir
    self.transform = None

  def __len__(self):

    return len([name for name in os.listdir(self.data_dir) if os.path.isfile(name)])

  def __getitem__(self, idx):

    img = torch.from_numpy(plt.imread(self.data_dir + "/images/" + str(idx) + ".jpg"))
    mask = torch.from_numpy(np.load(self.data_dir + "/masks/" + str(idx) + ".npy"))

    return img, mask

# Example usage

def main():

  dataset = ImageMaskDataset(str(os.path.dirname(os.getcwd())) + "/data")

  for i, (img, mask) in enumerate(dataset):
    
    if i > 10:
      break
      
    print(type(img), type(mask))

    plt.imshow(img)
    plt.show()

    plt.imshow(mask)
    plt.show()