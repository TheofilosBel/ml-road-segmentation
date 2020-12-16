import torch
import torch.nn as nn
from scripts.models.unet import UNet
from scripts.models.unet_mod import ModUnet


class ParUnets(nn.Module):
  """Implementation of a double unet model."""

  def __init__(self, in_channels, out_channels):
    super(ParUnets, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels    

    # Define 1st layer of Unet
    self.unet1 = UNet(in_channels, out_channels)

    # Define 1st layer of Unet
    self.unet2 = Unet(in_channels, out_channels, kernel_size=5, pad=2)

    # Output convolution
    self.out = nn.Sequential(
      nn.Conv2d(out_channels*2, 32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
    )

  def forward(self, input):
    x1 = self.unet1(input)
    x2 = self.unet2(input)    
    x = torch.cat((x1, x2), dim=1) # keep 2*2 channels changels    
    x = self.out(x)
    return x

        
  def pred(self, img, img_trans = None, normalized= True):
    ''' Predict the image & provide the correct transofmration'''
    self.eval()
    if img_trans != None:
      img = img_trans(img)
    img = img.unsqueeze(0)
    if torch.cuda.is_available():
      img = img.cuda()
    out = self.forward(img).cpu().detach()

    soft = nn.Softmax(dim=1)
    out_proba = soft(out)
    if normalized:
      return (out_proba[:,1] > 0.5).float()
    else:
      return out_proba