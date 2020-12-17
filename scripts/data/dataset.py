import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
import numpy as np
import math
import os

class SataDataset(Dataset):
  ''' 
    The Datalite Dataset loads all the data egarly and applies the 
    transformations
  '''
  def __init__(self, img_names,  # What data to include (strings)                 
                images_dir,      # Images path
                gt_images_dir,   # GroundTruths paht                 
                img_transform = None,
                gt_img_transform = None):
    '''
      Evrey transofrmation must create tensors. Also it must end up with a tensor
      like [SomeSize, C, H, W], where somesize can be 0.
    '''
    super().__init__()    
    self.imgs    = [mpimg.imread( os.path.join(images_dir, img_name) ) for img_name in img_names]
    self.gt_imgs = [mpimg.imread( os.path.join(gt_images_dir, img_name) )  for img_name in img_names]           
    
    if img_transform is not None:
      self.imgs = torch.stack( [img_transform(img) for img in self.imgs])
    if gt_img_transform is not None:
      self.gt_imgs = torch.stack( [gt_img_transform(gt_img) for gt_img in self.gt_imgs] )

      # Keep the format [Size, C, H, W] everytime
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
    
  def get_means_stds(self):
    '''
      Get the means and stds of all images in this dataset

      ### Returns:
        * A tuple like: (means, stds)
    '''
    if self.imgs == None or self.imgs.shape[0] == 0: 
      raise Exception("[ERR] Trying to get mean,std from an empty empty dataset.")

    # Compute the means and standard deviations for each of the 3 RGB channels
    # Remeber: self.imgs.shape = [Size, C, H, W]
    img_means = torch.mean(self.imgs, dim=[0,2,3]) 
    img_stds = torch.std(self.imgs, dim=[0,2,3])    
    
    # Return means and stds
    return img_means, img_stds

  def apply_standarization(self, img_means, img_stds):
    '''
      Apply a standarization to each images in the dataset

      ### Params:          
        * img_means & img_stds, should be the same size of the channels with the images of
        the dataset
    '''
    if self.imgs == None or self.imgs.shape[0] == 0: 
      raise Exception("[ERR] Trying to standarize on empty dataset.")  

    for idx in range(self.imgs.shape[0]):
      self.imgs[idx] = transforms.Normalize(img_means, img_stds)(self.imgs[idx])  


  def augment_with_gaus_blur(self, percentage = 0.35):
    '''
      Augment the dataset with Gausian Blur.
      Pick randomly a percentage of the dataset, add Gaussian blur and 
      append the new images to the dataset
    '''
    if self.imgs == None or self.imgs.shape[0] == 0: 
      raise Exception("[ERR] Trying to add blur on empty dataset.")      

    # Get a random permutation of indexes in the ds
    perms = np.random.permutation(len(self))[:math.floor(len(self) * percentage)]
    new_imgs = list()
    new_gt_imgs = list()
    for idx in perms:
      new_imgs.append(transforms.GaussianBlur(25)(self.imgs[idx].unsqueeze(0)).squeeze(0))
      new_gt_imgs.append(self.gt_imgs[idx])
        
    self.imgs = torch.cat((self.imgs, torch.stack(new_imgs)), 0)  
    self.gt_imgs= torch.cat((self.gt_imgs, torch.stack(new_gt_imgs)), 0)      
    