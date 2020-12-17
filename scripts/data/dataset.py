import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
import numpy as np
import math
import os

class SataDataset(Dataset):
  '''
    The Datalite Dataset loads all the data eagerly and applies the
    transformations.
  '''
  def __init__(self, img_names,  # What data to include (strings)
                images_dir,      # Images path
                gt_images_dir,   # GroundTruths paths
                img_transform = None,
                gt_img_transform = None):
    '''
      Create the dataset, by loading the data eagerly. It inputs the name of the images, and the dirs to load them from.

      ### Important NoteL
      Keep in mind that `image_names` should be the same for ground truth images and train images.

      ### Transformations:
      Every transformation must take as input a numpy array and create one tensor, the tensor must be of size [SomeSize, C, H, W], where somesize can be 0.
    '''
    super().__init__()
    self.imgs    = [mpimg.imread( os.path.join(images_dir, img_name) ) for img_name in img_names]
    self.gt_imgs = [mpimg.imread( os.path.join(gt_images_dir, img_name) )  for img_name in img_names]

    if img_transform is not None:
      self.imgs = torch.stack( [img_transform(img) for img in self.imgs])
    if gt_img_transform is not None:
      self.gt_imgs = torch.stack( [gt_img_transform(gt_img) for gt_img in self.gt_imgs] )

      # Keep the format [Size, C, H, W] whenever
      if len(self.imgs.shape) == 5:
        # Merege fist 2 dims
        dims = (-1,) + tuple(self.imgs.shape[2:])
        self.imgs = self.imgs.view(dims)

        # We imagine that the same must happen to the gt_imgs
        dims = (-1,) + tuple(self.gt_imgs.shape[2:])
        self.gt_imgs = self.gt_imgs.view(dims)


  def __len__(self):
    return self.imgs.shape[0]

  def __getitem__(self,index):
    img = self.imgs[index]
    gt_img = self.gt_imgs[index]
    return img, gt_img


  def augment_with_gaus_blur(self, percentage = 0.35):
    '''
      Augment the dataset with Gausian Blur.
      Pick randomly a % of the ds, add blur and
      append them to the ds as new images
    '''
    if self.imgs == None or self.imgs.shape[0] == 0: return

    # Get a random permutation of indexes in the ds
    perms = np.random.permutation(len(self))[:math.floor(len(self) * percentage)]
    new_imgs = list()
    new_gt_imgs = list()
    for idx in perms:
      new_imgs.append(transforms.GaussianBlur(25)(self.imgs[idx].unsqueeze(0)).squeeze(0))
      new_gt_imgs.append(self.gt_imgs[idx])

    self.imgs = torch.cat((self.imgs, torch.stack(new_imgs)), 0)
    self.gt_imgs= torch.cat((self.gt_imgs, torch.stack(new_gt_imgs)), 0)
