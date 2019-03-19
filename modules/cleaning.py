# cleaning
# Contains functions for cleaning images

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
import pycocotools
import collections


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


# Loads COCO data
def load_data():

    coco_data = torchvision.datasets.CocoDetection("/content/coco/val",
		"/content/coco/annotations/instances_val2017.json",
		torchvision.transforms.ToTensor())

    return coco_data


# Returns category ids, images (and ids) and annotations (and ids) of interest 
def get_interest(coco):

    super_categories = ['person', 'vehicle', 'animal']

    cat_ids = coco.getCatIds(supNms=super_categories)

    img_ids = []

    for cat_id in cat_ids:
        img_ids = img_ids + coco.getImgIds(catIds=cat_id)

    img_ids = list(set(img_ids))
    ann_ids = coco.getAnnIds(imgIds=img_ids)

    img_dict, ann_dict = get_dictionaries(coco, super_categories)

    return cat_ids, img_ids, ann_ids, img_dict, ann_dict


# Gets percentage of masking
def get_percent_mask(mask):

    num_occ = 0.0
    num_occ += (mask != 0).sum()
    num_tot = sum(len(item) for item in mask)

    return num_occ / num_tot


# Given annotations for object
# Returns a mask of up to only 4 prominent objects in the image
def get_prominent_mask(coco, anns):

    mask = np.zeros(np.array(coco.annToMask(anns[0])).shape)
    masks = []
    count = 0

    # order the annotations by percentage mask
    for i in range(len(anns)):

        obj_mask = coco.annToMask(anns[i])
        percent = get_percent_mask(obj_mask)
        masks.append((percent, obj_mask))

    # sort by percentage occupied
    masks.sort(key=itemgetter(0), reverse=True)

    # check if there exists a prominent object
    largest = masks[0][0]

    if largest < PROMINENT_PERCENT_THRESHOLD:
        return mask

    # filter with threshold
    masks_filtered = [mask_[1] for mask_ in masks if mask_[0] >= OTHER_OBJ_THRESH]

    for i in range(len(masks_filtered)):
        if count == MAX_PROMINENT_NUM:
            break

        mask += masks_filtered[i]
        count += 1

    # clip to form a binary mask
    mask = np.clip(mask, a_min=0, a_max=1)

    # count overall percentage
    all_percent = get_percent_mask(mask)

    # if its too large, zero out mask
    if all_percent > MAX_PERCENT:
        return mask * 0

    return mask


# Given an image
# Crops and resizes it to be resize_dim by resize_dim
# Fixes aspect ratio by padding to square
def resize(img, mask, resize_dim=IMG_SIZE):

    img = np.array(img)
    mask = np.array(mask)

    test = np.array(img)[0][0]
    is_rgb = test.size == 3
    type = 'RGB' if is_rgb else '1'

    # turn img into a PIL img
    im_pil = Image.fromarray(img, type)
    mask_im = Image.fromarray(np.uint8(255 * mask))

    # maintain aspect ratio and reduce to a dimension of resize_dim
    size = (resize_dim, resize_dim)

    im = im_pil.copy()
    msk = mask_im.copy()

    # copy image and corresponding mask into lower size
    im.thumbnail(size, Image.ANTIALIAS)
    msk.thumbnail(size, Image.ANTIALIAS)

    w, h = im.size

    # create new image with black background

    background = (0, 0, 0) if is_rgb else (0)

    square_im = Image.new(type, (resize_dim, resize_dim), background)
    square_msk = Image.new('1', (resize_dim, resize_dim))

    # paste in the original image to the square shaped one
    square_im.paste(im, (int((resize_dim - w) / 2), int((resize_dim - h) / 2)))
    square_msk.paste(msk, (int((resize_dim - w) / 2), int((resize_dim - h) / 2)))

    sq_msk_np = np.array(square_msk, dtype='float64')
    sq_msk_np = np.clip(sq_msk_np / 255, a_min=0, a_max=1)

    return np.array(square_im), sq_msk_np


# Given an image
# Resizes it to be resize_dim by resize_dim
# Does not maintain aspect ratio
def resize_square(img, mask, resize_dim=IMG_SIZE):

    img = np.array(img)
    mask = np.array(mask)

    test = np.array(img)[0][0]
    is_rgb = test.size == 3
    type = 'RGB' if is_rgb else '1'

    # turn img into a PIL img
    im_pil = Image.fromarray(img, type)
    mask_im = Image.fromarray(np.uint8(255 * mask))

    size = (resize_dim, resize_dim)
    im_pil = im_pil.resize(size, Image.ANTIALIAS)
    mask_im = mask_im.resize(size, Image.ANTIALIAS)

    mask_im_np = np.array(mask_im)
    mask_im_np = np.clip(mask_im_np, a_min=0, a_max=1)

    return np.array(im_pil), mask_im_np


# Given an image and annotation dictionaries
# Cleans the data to contain only prominent masks
# Returns cleaned images and masks as (image,mask) tuples of numpy arrays
def get_dataloader(coco, img_dict, ann_dict, size=None, crop=True, to_tensor=False, padding=False):

    transforms_ = transforms.Compose([transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])
    data = []
    count = 0

    # Randomly shuffle indices to access different images
    ind = np.arange(len(img_dict))
    random.shuffle(ind)

    ind = np.arange(len(img_dict))

    for i in ind:

        if count == size and size is not None:
            break

        print("Working on Image #", i)

        # generate prominent mask for object:
        mask = get_prominent_mask(coco, ann_dict[i])

        if np.sum(mask) != 0:

            img = img_dict[i]
            img = np.array(io.imread(img['coco_url']))

            if crop:

                if padding:
                    img, mask = resize(img, mask)
                else:
                    img, mask = resize_square(img, mask)

            if to_tensor:

                img = torch.from_numpy(np.transpose(img, [2, 0, 1]))
                img = transforms_(img.float())

                mask = torch.from_numpy(mask).unsqueeze(0)
                mask = mask.float()

            data.append((img, mask))
            count += 1

    if count != size and size is not None:
        print("Not enough data for size requested")

    return data


# Gets all images and annotations of interest
def get_dictionaries(coco, super_categories):

    # category id's for the super categories of interest
    categories = coco.getCatIds(supNms=super_categories)

    catIds_ = []
    imgIds = []
    annIds = []
    annDict = []

    for catId in categories:

        # get all images with at least this category in it
        imgIdVec = coco.getImgIds(catIds=catId)

        for item in imgIdVec:

            imgIds.append(item)

            # the ids for the annotation for this image, but only for the categories of interest
            annId = coco.getAnnIds(imgIds=item, catIds=categories, iscrowd=None)
            annDict.append(coco.loadAnns(annId))

    # create dictionary of images of interest
    imgDict = coco.loadImgs(imgIds)

    return imgDict, annDict


# Saves cleaned data onto disk
def generate_data(data):

    for i, point in enumerate(data):

        img, mask = point

        img = Image.fromarray(img)
        dir_img = "../data/images/" + str(i) + ".jpg"
        img.save(dir_img, 'JPEG')

        dir_mask = "../data/masks/" + str(i)
        np.save(dir_mask, mask)


# Sample usage of script
def main():

    print("Running main")

    coco = load_data().coco

    cat_ids, img_ids, ann_ids, img_dict, ann_dict = get_interest(coco)
    data = get_dataloader(coco, img_dict, ann_dict, size=None, to_tensor=False, padding=False)

    print("Total Images:", len(data))

    generate_data(data)
