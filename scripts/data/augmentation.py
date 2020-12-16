from scripts.utils.img import rotate, images_to_patches, patches_to_class
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import random

def add_noise(channel_stds, img, gt_img, patch):
  ''' img: Tensor [C,H,W]
      returns: Tensor [C,H,W]
  '''  
  if img.shape[0] > 3: raise TypeError('Dim 1 should be the channel') # sanity check
  gt = (gt_img > 0.25).float()
  height = img.shape[1]
  width = img.shape[2]
  
  # traverse both gt and img in this loop  
  for h in range(0,height, patch):
    for w in range(0,width, patch):
      proba = random.uniform(0,1)
      if torch.mean(gt[h:h+patch, w:w+patch]) > 0.8 and proba > 0.5: # road patch
        part = random.randint(0,3)
        p2 = int(patch/2)
        if part == 0:
          img[0,h:h+p2, w:w+patch] += random.uniform(0.9, 1.3) * channel_stds[0]
          img[1,h:h+p2, w:w+patch] += random.uniform(0.9, 1.3) * channel_stds[1]
          img[2,h:h+p2, w:w+patch] += random.uniform(0.9, 1.3) * channel_stds[2]
        elif part == 1:
          img[0,h:h+patch, w:w+p2] += random.uniform(0.9, 1.3) * channel_stds[0]
          img[1,h:h+patch, w:w+p2] += random.uniform(0.9, 1.3) * channel_stds[1]
          img[2,h:h+patch, w:w+p2] += random.uniform(0.9, 1.3) * channel_stds[2]
        elif part == 2:
          img[0,h+p2:h+patch:, w:w+patch] += random.uniform(0.9, 1.3) * channel_stds[0]
          img[1,h+p2:h+patch:, w:w+patch] += random.uniform(0.9, 1.3) * channel_stds[1]
          img[2,h+p2:h+patch:, w:w+patch] += random.uniform(0.9, 1.3) * channel_stds[2]
        elif part == 3:
          img[0,h:h+patch, w+p2:w+patch] += random.uniform(0.9, 1.3) * channel_stds[0]    
          img[1,h:h+patch, w+p2:w+patch] += random.uniform(0.9, 1.3) * channel_stds[1]    
          img[2,h:h+patch, w+p2:w+patch] += random.uniform(0.9, 1.3) * channel_stds[2]
  return img

def add_wholes(img, patch):
  ''' img: Tensor [H,W]
      returns: Tensor [H,W]
  '''    
  gt = (img > 0.25).float()
  height = img.shape[0]
  width = img.shape[21]
  
  # traverse both gt and img in this loop  
  for h in range(0,height, patch):
    for w in range(0,width, patch):
      proba = random.uniform(0,1)
      if torch.mean(gt[h:h+patch, w:w+patch]) > 0.8 and proba > 0.4: # road patch              
        img[0,h:h+patch, w:w+patch] = 0
  return img


# -------------------------
# Transformations for images
# -------------------------
def get_transformations_fcn(
  img_means= None, img_stds= None, \
  patch_size = 16 \
 ):
  '''
    Get a tuple with 2 transformations:
     * One transformation for the images
     * One transformation for the ground trouth images
    Adapt the transformations to the above parameters.

    The transformation assumes that the input will be only one numpy array 
    for the images [H,W,C] and [H,W] for the ground trouth images.
  '''  
  img_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda img: transforms.Normalize(img_means, img_stds)(img) if img_means != None else img),
    transforms.Lambda(lambda img: images_to_patches(img, patch_size, patch_size))
  ])
  img_gt_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda img: patches_to_class(images_to_patches(img, patch_size, patch_size)))    
  ])
  return img_trans, img_gt_trans

def get_transformations_unet(     \
  img_means= None, img_stds= None, \
  rotations= [45, -30, 25],  \
  center_crop_for_rot= 256,  \
 ):
  '''
    Get a tuple with 2 transformations:
     * One transformation for the images
     * One transformation for the ground trouth images
    Adapt the transformations to the above parameters.

    The transformation assumes that the input will be only one numpy array 
    for the images [H,W,C] and [H,W] for the ground trouth images.
  '''  
  img_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda img: transforms.Normalize(img_means, img_stds)(img) if img_means != None else img),
    transforms.Lambda(lambda img: rotate(img, rotations, center_crop_for_rot) if rotations != None else img)
  ])
  img_gt_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda img: rotate(img, rotations, center_crop_for_rot) if rotations != None else img),
    transforms.Lambda(lambda tensor: ((tensor.squeeze(1) if rotations != None else tensor.squeeze(0)) >0.25).long())
  ])
  return img_trans, img_gt_trans
