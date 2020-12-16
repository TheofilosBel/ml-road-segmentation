import torch
import torch.nn as nn

class ModUnet(nn.Module):
  """Implementation of the Modular Unet model."""

  def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, dropout=True, init_c=64):
    super(ModUnet, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.pad = pad
    multi_c = init_c
    drop_out_p = 0.1 if dropout else 0

    # Max pooling operation
    self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

    # Downsampling phase
    self.down_1 = self.double_conv(in_channels, multi_c, 0)
    self.down_2 = self.double_conv(multi_c, multi_c*2, 0)
    self.down_3 = self.double_conv(multi_c*2, multi_c*4, 0)
    self.down_4 = self.double_conv(multi_c*4, multi_c*8, 0)
    self.down_5 = self.double_conv(multi_c*8, multi_c*16, 0)

    # Upsampling phase
    self.up_trans_1 = self.up_trans(multi_c*16, multi_c*8)
    self.up_1 = self.double_conv(multi_c*16, multi_c*8, drop_out_p * 3)

    self.up_trans_2 = self.up_trans(multi_c*8, multi_c*4)
    self.up_2 = self.double_conv(multi_c*8, multi_c*4, drop_out_p * 3)
    
    self.up_trans_3 = self.up_trans(multi_c*4, multi_c*2)
    self.up_3 = self.double_conv(multi_c*4, multi_c*2, drop_out_p * 2)
    
    self.up_trans_4 = self.up_trans(multi_c*2, multi_c)
    self.up_4 = self.double_conv(multi_c*2, multi_c, drop_out_p * 2)

    # Output convolution
    self.out = nn.Conv2d(multi_c, out_channels, kernel_size=1)

  def up_trans(self, in_channels, out_channels):
    """Upsampling operation"""
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

  def double_conv(self, in_channels, out_channels, drop_out_p):
    """Double convolution (each followed by a batch normalization and a RELU)"""
    if drop_out_p == 0:
      return nn.Sequential(
        # 1st convolution    
        nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.pad, padding_mode='reflect'),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),       
        # 2nd convolution    
        nn.Conv2d(out_channels, out_channels, kernel_size=self.kernel_size, padding=self.pad, padding_mode='reflect'),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
      )
    else:
      return nn.Sequential(
        # 1st convolution    
        nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.pad, padding_mode='reflect'),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(drop_out_p),
        # 2nd convolution    
        nn.Conv2d(out_channels, out_channels, kernel_size=self.kernel_size, padding=self.pad, padding_mode='reflect'),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
      )

  def forward(self, input):
    # Downsampling part (keep the outputs for the concatenation in the upsampling stage)
    down_x1 = self.down_1(input)
    down_x2 = self.down_2(self.max_pool_2x2(down_x1))
    down_x3 = self.down_3(self.max_pool_2x2(down_x2))
    down_x4 = self.down_4(self.max_pool_2x2(down_x3))
    down_x5 = self.down_5(self.max_pool_2x2(down_x4))

    # Upsampling part (concatenate outputs)
    x = self.up_trans_1(down_x5)
    x = self.up_1(torch.cat([down_x4, x], dim=1))

    x = self.up_trans_2(x)    
    x = self.up_2(torch.cat([down_x3, x], dim=1))

    x = self.up_trans_3(x)
    x = self.up_3(torch.cat([down_x2, x], dim=1))

    x = self.up_trans_4(x)
    x = self.up_4(torch.cat([down_x1, x], dim=1))

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