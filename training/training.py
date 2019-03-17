# Library Dependencies

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import os
import shutil
import time
import re
import sys
import skimage.io as io
import random
from operator import itemgetter
from PIL import Image
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import pycocotools
import collections
import os


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


def main():

    dataset = ImageMaskDataset(str(os.path.dirname(os.getcwd())) + "/data")

    for img, mask in dataset:

        plt.imshow(img)
        plt.show()

        plt.imshow(mask)
        plt.show()


if __name__ == "__main__":
    main()
