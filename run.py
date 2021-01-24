# Import python modules
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from typing import List
import math
import random
import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

from scripts.data.dataset import SataDataset
from scripts.data.preprocess import get_means_stds
from scripts.data.augmentation import get_transformations_fcn, get_transformations_unet
from scripts.models.fcn import FCN
from scripts.models.unet import UNet
from scripts.models.res_unet import ResUNet
from scripts.models.trainer import train
from scripts.utils.submission import masks_to_submission
from scripts.utils.img import crops_to_img

################
# IMAGE PATHS  #
################

# Data paths
data_root = './data/unzip/training/'
all_img_names = os.listdir(os.path.join(data_root,'images/'))
imgs_path = os.path.join(data_root,'images/')
gt_imgs_path = os.path.join(data_root,'groundtruth/')

# Test imgs path
sub_imgs ='data/unzip/test_set_images/'
all_sub_img_names = os.listdir(sub_imgs)
all_imgs_paths_map = [(f'{sub_imgs}{name}/{name}.png', name.split('_')[1]) for name in all_sub_img_names]

# The path for the submission file
submission_file_name = 'submission.csv'

############
# Training #
############

# Train test split
train_set, test_set = train_test_split(all_img_names, test_size=0.1)

# Compute the means and standard deviations for each of the 3 RGB channels
img_means, img_stds = get_means_stds(train_set, imgs_path)

# Get tranformations
toPil_trans = transforms.ToPILImage() # Used for printing tensors
model = None
use_unet = True

# UNET: Load transformations for the 2 datasets
img_trans, gt_img_trans = get_transformations_unet(img_means, img_stds)
val_img_trans, val_gt_img_trans = get_transformations_unet(img_means, img_stds)

# FCN: Load transformations for the 2 datasets
# img_trans, gt_img_trans = get_transformations_fcn(patch_size=16)
# val_img_trans, val_gt_img_trans = get_transformations_fcn(patch_size=16)
# model = FCN(3, 2, 1024)

# Create Datasets
train_ds = SataDataset(train_set, imgs_path, gt_imgs_path, img_trans, gt_img_trans)
val_ds = SataDataset(test_set, imgs_path, gt_imgs_path, val_img_trans, val_gt_img_trans)

# Create data loaders that break data to batches
print(f'Loading data...')
train_dl = DataLoader(dataset=train_ds, batch_size=5, shuffle=True, num_workers=0)
val_dl = DataLoader(dataset=val_ds, batch_size=1, shuffle=True, num_workers=0)

# Define the model to use
print('Starting training...')
model = UNet(3, 2)
# model = ResUNet(3, 2)
# model = FCN(3, 2, 1024)

# Define optimizer parameters & loss function
learning_rate = 0.01
optimizer = opt.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.001)
loss_func = nn.CrossEntropyLoss()

# To cuda if vailable
if torch.cuda.is_available():
  model = model.cuda()
  loss_func = loss_func.cuda()

# Define epochs and train the model
epochs = 100
train_losses, val_losses, metrics = train(epochs, model, loss_func, optimizer, train_dl, val_dl, 5, [f1_score, recall_score, precision_score, accuracy_score])


#####################
# Create Submission #
#####################
print('Creating submissions.scv...')
model.eval()
predicted_map = list()
with torch.no_grad():
  for path, num in all_imgs_paths_map:
    img = mpimg.imread(path)  # load
    prediction = model.pred(img, val_img_trans)
    predicted_map.append((prediction, int(num)))

masks_to_submission(submission_file_name, predicted_map, 0.25)