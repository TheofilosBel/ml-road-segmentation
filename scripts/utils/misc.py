######################
# OLD UNET (REFLECT) #
######################

# Unet tranfomartions
# Define Transofmratio for the Dataset
angles = [45, -45, 90, -90]

def rotate(img, angles):
  rotations = list()
  rotations.append(img)
  for angle in angles:
    t = transforms.Compose([
      transforms.Pad(82, padding_mode='reflect'),
      transforms.Lambda(lambda tensor: transforms.functional.rotate(tensor, angle)),  
      transforms.CenterCrop(400)
    ])
    rotations.append( t(img) )
  return torch.stack( rotations )

t_x = transforms.Compose([
  transforms.ToTensor(),
])
t_y = transforms.Compose([
  transforms.ToTensor(),
  transforms.Lambda(lambda tensor: tensor.squeeze(0)),
  transforms.Lambda(lambda tensor: (tensor>0.25).long())
])


rotate_trans_x = transforms.Compose([
  transforms.ToTensor(),             
  transforms.Lambda(lambda img: rotate(img, angles))  
])
rotate_trans_y = transforms.Compose([  
  transforms.ToTensor(),
  transforms.Lambda(lambda img: rotate(img, angles)),
  transforms.Lambda(lambda tensor: (tensor.squeeze(1)>0.25).long())
])

# Create 2 Data loaders for each case
train_dl = DataLoader(dataset = SataDataset(train_set, imgs_path, gt_imgs_path, rotate_trans_x, rotate_trans_y), batch_size=5, shuffle=True, num_workers=0)
test_dl = DataLoader(dataset = SataDataset(test_set, imgs_path, gt_imgs_path, t_x, t_y), batch_size=1, shuffle=True, num_workers=0)


####################################
# DOUBLE UNET with cat of real img #
####################################

# Define transofrmations
toPil_trans = transforms.ToPILImage()

# Load the Unet to use its output as gt
unet = UNet(3,2)
load(unet, name_specifier='256_3angles_norm', path='../../road_seg/cached_models')

# Define Tranformations for the Dataset
img_trans = transforms.Compose([
  transforms.ToTensor(),
  transforms.Lambda(lambda tensor: torch.cat((tensor, unet.pred(tensor, normalized=False)[0,1,:].unsqueeze(0)), 0))  
])
img_gt_trans = transforms.Compose([
  transforms.ToTensor(),
  transforms.Lambda(lambda tensor: tensor.squeeze(0)),
  transforms.Lambda(lambda tensor: (tensor>0.25).long())
])


# Create 2 Data loaders for each case
train_dl = DataLoader(dataset = SataDataset(train_set, imgs_path, gt_imgs_path, img_trans, img_gt_trans), batch_size=5, shuffle=True, num_workers=0)
test_dl = None #DataLoader(dataset = SataDataset(test_set, imgs_path, gt_imgs_path, img_trans, img_gt_trans), batch_size=1, shuffle=True, num_workers=0)

###########################
# UNET WITH NORMALIZATION #
###########################

# Unet tranfomartions
# Define Transofmratio for the Dataset
angles = [45, -30, 25]
center_crop_for_rot = 256 # math.floor((img.shape[1]/2) * math.sqrt(2))  # Get H or W, they are the same 

def rotate(img, angles, center_crop_for_rot):
  rotations = list()
  rotations.extend(transforms.FiveCrop(center_crop_for_rot)(img))
  for angle in angles:
    t = transforms.Compose([      
      transforms.Lambda(lambda tensor: transforms.functional.rotate(tensor, angle)),  
      transforms.CenterCrop(center_crop_for_rot)
    ])
    rotations.append( t(img) )
  return torch.stack( rotations )

# Validation tranforms
test_img = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(img_means, img_stds)
])
test_gt_img = transforms.Compose([
  transforms.ToTensor(),  
  transforms.Lambda(lambda tensor: tensor.squeeze(0)),
  transforms.Lambda(lambda tensor: (tensor>0.25).long())
])

# Train transforms
rotate_trans_x = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(img_means, img_stds),
  transforms.Lambda(lambda img: rotate(img, angles, center_crop_for_rot))  
])
rotate_trans_y = transforms.Compose([  
  transforms.ToTensor(),
  transforms.Lambda(lambda img: rotate(img, angles, center_crop_for_rot)),
  transforms.Lambda(lambda tensor: (tensor.squeeze(1)>0.25).long())
])

# Create 2 Data loaders for each case
train_dl = DataLoader(dataset = SataDataset(train_set, imgs_path, gt_imgs_path, rotate_trans_x, rotate_trans_y), batch_size=5, shuffle=True, num_workers=0)
test_dl = DataLoader(dataset = SataDataset(test_set, imgs_path, gt_imgs_path, test_img, test_gt_img), batch_size=1, shuffle=True, num_workers=0)
