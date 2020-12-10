import torch
import torch.nn as nn

class UNet(nn.Module):
  """Implementation of the UNet model."""

  def __init__(self, in_channels, out_channels):
    super(UNet, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels

    # Max pooling operation
    self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

    # Downsampling phase
    self.down_1 = self.double_conv(in_channels, 64)
    self.down_2 = self.double_conv(64, 128)
    self.down_3 = self.double_conv(128, 256)
    self.down_4 = self.double_conv(256, 512)
    self.down_5 = self.double_conv(512, 1024)

    # Upsampling phase
    self.up_trans_1 = self.up_trans(1024, 512)
    self.up_1 = self.double_conv(1024, 512)

    self.up_trans_2 = self.up_trans(512, 256)
    self.up_2 = self.double_conv(512, 256)
    
    self.up_trans_3 = self.up_trans(256, 128)
    self.up_3 = self.double_conv(256, 128)
    
    self.up_trans_4 = self.up_trans(128, 64)
    self.up_4 = self.double_conv(128, 64)

    # Output convolution
    self.out = nn.Conv2d(64, out_channels, kernel_size=1)

  def double_conv(self, in_channels, out_channels):
    """Double convolution (each followed by a batch normalization and a ReLU)"""
    return nn.Sequential(
      # 1st convolution
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      # 2nd convolution
      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
    )

  def up_trans(self, in_channels, out_channels):
    """Upsampling operation"""
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

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