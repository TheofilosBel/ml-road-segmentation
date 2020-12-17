import os
import numpy as np
import matplotlib.image as mpimg
import torch


def patch_to_label(patch, foreground_threshold):
  ''' Assign a label to a patch '''
  df = torch.mean(patch)
  if df.item() > foreground_threshold:
    return 1
  else:
    return 0


def mask_to_submission_strings(im, img_num, foreground_threshold):
  """Reads a single image and outputs
      the strings that should go into the submission file

      # Params:
      * im: tensor of size [C,H,W]
  """
  patch_size = 16
  for j in range(0, im.shape[2], patch_size):
    for i in range(0, im.shape[1], patch_size):
      patch = im[:, i:i + patch_size, j:j + patch_size]
      label = patch_to_label(patch, foreground_threshold)
      yield("{:03d}_{}_{},{}".format(img_num, j, i, label))

def masks_to_submission(submission_filename, image_map, foreground_threshold):
  """Converts images into a submission file"""
  with open(submission_filename, 'w') as f:
    f.write('id,prediction\n')
    for img, img_num in image_map:
      f.writelines('{}\n'.format(s) \
        for s in mask_to_submission_strings(img, img_num, foreground_threshold))


def img_to_patches(im, foreground_threshold):
  """Reads a single image and outputs
      the strings that should go into the submission file

      # Params:
      * im: tensor of size [C,H,W]
  """
  patched_img = torch.zeros((im.shape[1],im.shape[2]))
  patch_size = 16
  for j in range(0, im.shape[2], patch_size):
    for i in range(0, im.shape[1], patch_size):
      patch = im[:, i:i + patch_size, j:j + patch_size]
      label = patch_to_label(patch, foreground_threshold)
      patched_img[i:i + patch_size, j:j + patch_size] = label
  return patched_img






