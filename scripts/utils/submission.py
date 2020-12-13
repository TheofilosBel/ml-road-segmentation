import os
import numpy as np
import matplotlib.image as mpimg
import re
import torch


# assign a label to a patch
def patch_to_label(patch, foreground_threshold):    
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



def create_submissions(model, file_name):
  net.eval()
  predicted_map = list()
  for path, num in all_imgs_paths_map:
    img = mpimg.imread(path)  # load
    input = t_x(img)
    if torch.cuda.is_available():
      input = input.cuda()
    out = net(input.unsqueeze(0)).cpu().detach()
    a = nn.Softmax(dim=1)
    out_proba = a(out)
    prediction = (out_proba[:,1] > 0.5).float()
    predicted_map.append((prediction, int(num)))

  masks_to_submission(file_name, predicted_map, 0.25)



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
      





