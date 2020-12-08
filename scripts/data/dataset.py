import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader
from scripts.utils.img import to_class
import os

class SataDataset(Dataset):
  def __init__(self, img_names,       # What data to include (strings)                 
                images_dir,    # Images path
                gt_images_dir, # GroundTruths paht                 
                transform = None):
    super().__init__()
    self.data = img_names
    self.imgs    = [mpimg.imread( os.path.join(images_dir, img_name) ) for img_name in img_names]
    self.gt_imgs = [mpimg.imread( os.path.join(gt_images_dir, img_name) )  for img_name in img_names]           
    
    if transform is not None:      
      self.imgs = [transform(img) for img in self.imgs]
      self.gt_imgs = [to_class( transform(gt_img) ) for gt_img in self.gt_imgs]      

      # discard patches that are black
      # filtered_imgs = []
      # for img in self.imgs:
      #   filtered_patches = [patch for patch in img if patch.mean() > 0.2]
      #   filtered_imgs.append( torch.stack(filtered_patches) )
      # self.imgs = filtered_imgs
    
  def __len__(self):    
    return len(self.data)
    
  def __getitem__(self,index):    
    img = self.imgs[index]
    gt_img = self.gt_imgs[index]    
    return img, gt_img