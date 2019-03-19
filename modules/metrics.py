# metrics.py
# Contains functions for performance evaluation


import torch
import numpy as np


# Given an object mask
# Returns percentage of image occupied by object
def getPercentMask(obj_mask):
	num_occ = 0.0
	num_occ += (obj_mask != 0).sum()
	num_tot = sum(len(item) for item in obj_mask)

	return num_occ/num_tot


# Given prediction mask and target masks
# Compute accuracy metric: IoU
def calculateIoU(pred, target):
	true_mask = target.squeeze(1).detach().numpy()
	pred_mask = pred.squeeze(1).detach().numpy()

	intersect = np.sum((true_mask*pred_mask), axis=(1,2)) 
	union = np.sum(np.clip(true_mask + pred_mask, a_min=0, a_max=1), axis=(1,2))

	# Tensor of IoU for each image
	iou = intersect/union

	# Sum IoU over all images in batch
	return np.sum(iou)  # TODO: This is a summation of of IoUs when multiple images are fed


# Given a model and dataset of (image, mask) tuples, calculate overall IoU
def calculateTotalIoU(model, data):
	data_loader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=False)

	total_iou = 0
	num_data = 0

	for i, batch in enumerate(data_loader):
		img, target = batch
		pred = model(img)
		iou = calculateIoU(pred, target)
		total_iou += iou

		num_data += img.shape[0]

	# Average IoU on data set  
	return total_iou/num_data


# Computes average loss over dataset
def getLoss(model, data, criterion):
	data_loader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=False)

	total_loss = 0
	num_data = 0

	for i, batch in enumerate(data_loader):
		img, target = batch
		pred = model(img)
		loss = criterion(pred, target)
		total_loss += loss
		num_data += img.shape[0]

	return total_loss/num_data
