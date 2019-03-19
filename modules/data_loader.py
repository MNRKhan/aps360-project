# data_loader.py
# Contains functions for loading coco dataset


import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F


# Returns (image, target) tensor tuples
def loadData():
	cocoData = torchvision.datasets.CocoDetection("/content/coco/val",
		"/content/coco/annotations/instances_val2017.json",
		torchvision.transforms.ToTensor())
	return cocoData


# Returns the image dictionary and corresponding annotation id dictionary 
# Containing images of the classes of interest (string specification, not id)
def generateSelImgDict(cocoData, interestSuperCats):
	# All categories
	#cats = (cocoData.coco).loadCats((cocoData.coco).getCatIds())

	# Get category ids for the super categories of interest
	interestCatIds = (cocoData.coco).getCatIds(supNms=interestSuperCats)

	imgIds = []
	annDict = []

	# Extract images with at least one of the categories of interest in them
	for catId in interestCatIds:
		#catId = cocoData.coco.getCatIds(catNms = cat['name'])

		# Get all images with at least this category in it
		imgIdVec = (cocoData.coco).getImgIds(catIds=catId)

		# For each image, add them to the id list along with its annotations
		for item in imgIdVec:
			imgIds.append(item)

			# Get ids for the annotation for this image for the categories of interest
			annId = cocoData.coco.getAnnIds(imgIds=item, catIds=interestCatIds, iscrowd=None)
			annDict.append(cocoData.coco.loadAnns(annId))

	# Create dictionary of images of interest
	imgDict = (cocoData.coco).loadImgs(imgIds)

	return imgDict, annDict

