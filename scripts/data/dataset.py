import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader
from scripts.utils.img import to_class
import os
import torch

class SataDataset(Dataset):
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