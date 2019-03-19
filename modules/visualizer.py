# visualizer.py
# Contains functions for image visualization and modification


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


# Given annotations for object
# Returns a mask of up to only 4 prominent objects in the image
def getProminentMask(annotations, cocoData):
	mask = np.zeros(np.array(cocoData.coco.annToMask(annotations[0])).shape)
	masks = []
	count = 0

	# Order the annotations by percentage mask
	for i in range(len(annotations)):
		obj_mask = cocoData.coco.annToMask(annotations[i])
		percent = getPercentMask(obj_mask)
		masks.append((percent, obj_mask))

	# Sort by percentage occupied
	masks.sort(key=itemgetter(0), reverse=True)

	# Check if there exists a prominent object
	largest = masks[0][0]

	# Return the empty mask if not prominent objects
	if(largest < PROMINENT_PERCENT_THRESHOLD):
		return mask

	# Extract only the relevant masks
	# Since there is at least one prominent object, there can be smaller auxilary objects
	masks_filtered = [mask_[1] for mask_ in masks if mask_[0] >= OTHER_OBJ_THRESH]

	for i in range(len(masks_filtered)):
		if (count == MAX_PROMINENT_NUM):
			break

		mask += masks_filtered[i]
		count += 1

	# Clip to form a binary mask
	mask = np.clip(mask, a_min=0, a_max=1)

	# Calculate overall percentage
	all_percent = getPercentMask(mask)

	# Images which have some background is desired
	# if too large, zero out mask
	if (all_percent > MAX_PERCENT):
		return mask * 0

	return mask


# Given an image
# Crops and resizes it to be resize_dim by resize_dim
# Fixes aspect ratio by padding to square
def cropResizeSquare(img, mask, resize_dim=224):
	img = np.array(img)
	mask = np.array(mask)

	# Convert image to a PIL image
	img_pil = Image.fromarray(img, 'RGB')
	img_mask = Image.fromarray(np.uint8(255*mask)) 
	
	# Maintain aspect ratio and reduce to a dimension of resize_dim
	size = (resize_dim, resize_dim)

	im = img_pil.copy()
	msk = img_mask.copy()

	# Copy image and corresponding mask into lower size
	im.thumbnail(size, Image.ANTIALIAS)
	msk.thumbnail(size, Image.ANTIALIAS)

	w,h = im.size

	# Create new image with black background
	square_im = Image.new('RGB', (resize_dim, resize_dim), (0,0,0))
	square_msk = Image.new('1', (resize_dim, resize_dim))

	# Paste original image onto the square shaped image
	square_im.paste(im, (int((resize_dim-w)/2), int((resize_dim-h)/2)))
	square_msk.paste(msk, (int((resize_dim-w)/2), int((resize_dim-h)/2)))

	sq_msk_np = np.array(square_msk)
	sq_msk_np = np.clip(sq_msk_np, a_min=0, a_max=1)
	
	return np.array(square_im), sq_msk_np


# Given an image
# Resizes it to be resize_dim by resize_dim
# Does not maintain aspect ratio
def resizeSquare(img, mask, resize_dim=224):
	img = np.array(img)
	mask = np.array(mask)

	# Convert image to a PIL image
	im_pil = Image.fromarray(img, 'RGB')
	mask_im = Image.fromarray(np.uint8(255*mask)) 

	size = (resize_dim, resize_dim)
	im_pil = im_pil.resize(size, Image.ANTIALIAS)
	mask_im = mask_im.resize(size, Image.ANTIALIAS)

	mask_im_np = np.array(mask_im)
	mask_im_np = np.clip(mask_im_np, a_min=0, a_max=1)

	return np.array(im_pil), mask_im_np


# Given an image and annotation dictionaries
# Cleans the data to contain only prominent masks
# Returns cleaned images and masks as (image,mask) tuples of numpy arrays
def dataParseProminent(cocoData, imgDict, annDict, size, just_resize=False, 
						crop=True, toTensor=False, path='None'):
	transforms_ = transforms.Compose([transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])
	data = []
	count = 0
	
	# Randomly shuffle indices to access different images
	ind = np.arange(len(imgDict))
	random.shuffle(ind)

	for i in ind:
		# Stop when size is reached
		if(count == size):
			break
		
		# Generate prominent mask for object
		mask = getProminentMask(annDict[i], cocoData)

		# if entire mask is not 0 (has some prominent objects in the mask)
		if np.sum(mask)!=0:
			imgInf = imgDict[i]
			img = np.array(io.imread(imgInf['coco_url']))

			if(just_resize == True):
				img,mask = resizeSquare(img, mask)
			elif (crop==True):
				img,mask = cropResizeSquare(img, mask)

			if (toTensor == True):
				# Transpose to [C,H,W] format for convolution
				img = torch.from_numpy(np.transpose(img, [2,0,1]))
				img = transforms_(img.float())

				mask = torch.from_numpy(mask).unsqueeze(0)
				mask = mask.float()

			data.append((img, mask))
			count += 1

	if (count != size):
		print('Warning: Not enough data for size requested')

	return data


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


