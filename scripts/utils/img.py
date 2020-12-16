import torch
import torchvision.transforms as transforms

#
# Patch handling
#

def images_to_patches(im, w, h):
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

def crops_to_img(im, init_w, init_h):
  ''' 
  From crops get an image
  ## Params:
    * im: Tensors of shape [Patches, C, H, W]
    * the h,w before cropping the im
  ## Returns:
    A tensor of shape [C,H,W]
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

def crops_to_label(im, init_w, init_h):
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


def patches_to_class(tensor, pixel_thres=0.25, patch_thres=0.25):
  ''' 
    ## Params:
     * `tensor`: [PatchNum, 1, HPatch, WPatch]
  '''    
  road_pixels = (tensor > pixel_thres).int()  
  pixels_in_patch = tensor.shape[2] * tensor.shape[3]  
  sum_per_patch = road_pixels.sum( (2,3) )   
  pathes_to_label = (sum_per_patch / pixels_in_patch) > patch_thres # Patches with 25% roads are roads  
  return pathes_to_label.long().view(-1)

#
# Rotates 
#

def rotate(img, angles, center_crop_for_rot):
  '''
    Five crops the image. Then rotate the image and Center crop it for each angle
    ## Params:
      * `img`: a tensor of size [C,H,W]
      * `angles`: a list of ints with the angles
      * `center_crop_for_rot`: The size of the crop to apply after the rotations 
        also in the FiveCrop
    ## Returns:
      * `tensor`: size of [5+len(angles), C, H, W]
  '''
  rotations = list()
  rotations.extend(transforms.FiveCrop(center_crop_for_rot)(img))    
  for angle in angles:
    t = transforms.Compose([      
      transforms.Lambda(lambda tensor: transforms.functional.rotate(tensor, angle)),  
      transforms.CenterCrop(center_crop_for_rot)
    ])
    rotations.append( t(img) )
  return torch.stack( rotations )


#
# Other ideas
#

def boundry_map_to_coordinates(boundry_map_h):
  '''Translate the map to a list of coordinates. 
     Map should be like: {h: list_of_w}
  '''
  coordinates = list()
  for h, list_w in boundry_map_h.items():
    for w in list_w:
      coordinates.append( (h,w) )
  return coordinates


def coordinates_to_img(coordinates, img_to_copy):
  '''img_to_copy: must be size [H, W]
  '''  
  img = torch.zeros(img_to_copy.shape)  
  for h,w in coordinates:
    img[h, w] = 1.0  
  return img


def find_boundries(img, gt_img, window):
  ''' Find `window` pixels in img that are right near to roads 

      ## Params:
       * img: tensor size [C, H, W]
       * gt_img: tensor size [H, W]

      ## Returns:
       * A map like this: {h: list_of_w}, where each (h,w) is a boundry
  '''
  if window < 1: raise TypeError('window must be >1')
  if img.shape[0] > 10: raise TypeError('Dim 1 should be the channel') # sanity check
  gt = (gt_img > 0.25).int()
  height = img.shape[1]
  width = img.shape[2]
  
  # Store boundry elements in a dict
  boundry_map_h = dict()  

  # traverse both gt and img in this loop  
  for h in range(0,height):
    for w in range(0,width):
      # if its a road pixel
      found_bound = False
      if gt[h, w] == 1:
        # eplore all non read pixels near it:
        # a) height exploration
        for dir_h in [-1,1]:
          for win in range(1, window + 1):
            new_h = h + dir_h * win
            if new_h < 0 or new_h >= height: continue  # new_h is not in pic            
            if gt[new_h, w] == 0: # return non road elements only              
              boundry_map_h.setdefault(new_h, set()).add(w) # added only if not present              
              found_bound = True
              
        # b) width exploration
        for dir_w in [-1,1]:
          for win in range(1, window + 1):
            new_w = w + dir_w * win            
            if new_w < 0 or new_w >= width: continue
            if gt[h, new_w] == 0: # return non road elements only              
              boundry_map_h.setdefault(h, set()).add(new_w) # added only if not present    
              found_bound = True              

  # Return the a tensor and the last_road_coordinates
  return coordinates_to_img(boundry_map_to_coordinates(boundry_map_h), gt_img)
