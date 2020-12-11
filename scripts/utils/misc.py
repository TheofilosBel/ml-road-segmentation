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