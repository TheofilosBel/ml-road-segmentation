import torch
import torchvision.transforms as transforms

# Crops a tensor
def img_crop(im, w, h):
  ''' 
    ## Params:
     * im: Tensors of shape [C, H, W]

    ## Returns:
     * Tensor of size [PatchesNum, C, H, W]
  '''
  list_patches = []
  imgheight = im.shape[1]
  imgwidth = im.shape[2]
  for i in range(0,imgheight,h):
    for j in range(0,imgwidth,w):
      patch = transforms.functional.crop(im, i, j, h, w)
      list_patches.append( patch.unsqueeze(0) )
  return torch.cat( list_patches )

def crop_to_img(im, init_w, init_h):
  ''' im: Tensors of shape [Patches, C, H, W]
  '''
  img = torch.zeros((3, init_h, init_w))
  crop_height = im.shape[2]
  crop_width = im.shape[3]
  idx = 0
  for i in range(0, init_h, crop_height):    
    for j in range(0, init_h, crop_width):
      img[:, i:i+crop_height, j:j+crop_width] = im[idx]
      idx+=1  
  return img

def crop_to_label(im, init_w, init_h):
  ''' im: Tensors of shape [Patches]
  '''
  img = torch.zeros((1, init_h, init_w))
  crop_height = 16
  crop_width = 16
  idx = 0
  for i in range(0, init_h, crop_height):    
    for j in range(0, init_h, crop_width):
      img[:, i:i+crop_height, j:j+crop_width] = im[idx]
      idx+=1  
  return img

# Crop to class
def to_class(tensor):
  ''' tensor: [PatchNum, 1, HPatch, WPatch]
  '''    
  road_pixels = (tensor > 0.25).int()  
  pixels_in_patch = tensor.shape[2] * tensor.shape[3]  
  sum_per_patch = road_pixels.sum( (2,3) )   
  pathes_to_label = (sum_per_patch / pixels_in_patch) > 0.25 # Patches with 25% roads are roads  
  return pathes_to_label.long().view(-1)