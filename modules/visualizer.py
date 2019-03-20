# visualizer.py
# Contains functions for image visualization


import matplotlib.pyplot as plt
import numpy as np
import random
import skimage.io as io
import torch
from operator import itemgetter
from PIL import Image
from torchvision import datasets, models, transforms

from metrics import getPercentMask, calculateIoU


# Minimum fraction of an image the object must occupy in order to be considered prominent
PROMINENT_PERCENT_THRESHOLD = 0.3

# Extract images with one very prominent object and other possible smaller objects
OTHER_OBJ_THRESH = 0.1

# Maximum fraction of an image the object must occupy in order to be considered prominent
MAX_PERCENT = 0.9

# Default input dimensions
IMG_SIZE = 224

# Maximum number of objects that are considered to be prominent
MAX_PROMINENT_NUM = 4


# Displays an image
def imshow(img, show_axis=False, save=False, save_path=None):
	if not show_axis:	plt.axis('off')
	plt.imshow(img)
	if save:	plt.savefig(save_path)
	plt.show()
	plt.clf()


# Returns bit mask for objects of interset in image
def getBitMask(annotations, cocoData):
	mask = cocoData.coco.annToMask(annotations[0])

	# Create conglomerate mask over all objects in image
	for i in range(len(annotations)):
		mask = mask | cocoData.coco.annToMask(annotations[i])

	#imshow(mask)
	return mask


# Returns masked image
def getMaskedImg(img, mask):
	mask_arr = np.array(mask)

	# Reshape to give 3rd axis for broadcasting to 3 channels
	mask_arr = np.expand_dims(mask_arr, axis=-1)
	masked_img = np.array(img)

	masked_img = masked_img * mask_arr

	return masked_img


# Given a tensor of images in NCHW format, converts to numpy images
def tensorToNpImg(tensor, img_type='mask'):
	image = tensor.detach().numpy()

	# Re-normalize for imshow plotting
	if(img_type != 'mask'):
		image = image/255

	image = np.transpose(image, [1,2,0])
	return image.squeeze()


def thresholdProbMask(prob_mask, threshold=0.5):
	prob_mask[prob_mask>threshold] = 1
	prob_mask[prob_mask<=threshold] = 0
	return prob_mask


# Given model, input image, and target mask
# Evaulates output mask using model and displays against target
def extractProminent(model, img, target):
	plt.figure()
	plt.subplot(1,3,1)
	plt.imshow(tensorToNpImg(img, 'img'));  plt.axis('off')

	plt.subplot(1,3,2)
	plt.imshow(tensorToNpImg(target));  plt.axis('off')

	res = model(img.unsqueeze(0).float())
	plt.subplot(1,3,3)
	generatedMask = thresholdProbMask(tensorToNpImg(res.squeeze(0)))
	plt.imshow(generatedMask);  plt.axis('off')
	
	print("IoU:", calculateIoU(res, target))


# Plots curve for given train and validation arrays
# ctype={'Accuracy","Loss"}
def plotCurve(train_val, valid_val, num_epochs, ctype):
	plt.title('Train vs Validation {}'.format(ctype))
	plt.plot(range(num_epochs), train_val, label='Train')
	plt.plot(range(num_epochs), valid_val, label='Validation')
	plt.xlabel('Epoch')
	plt.ylabel(ctype)
	plt.legend(loc='best')
	plt.show()


def plotPerformance(train_loss, valid_loss, train_acc, valid_acc, num_epochs):

	# Plot loss curves
	plotCurve(train_loss, valid_loss, num_epochs, ctype = 'Loss')

	# Plot accuracy curves
	plotCurve(train_acc, valid_acc, num_epochs, ctype = 'IoU') 


