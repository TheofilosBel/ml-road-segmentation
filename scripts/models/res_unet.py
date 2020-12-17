import torch
import torch.nn as nn

class ResUNet(nn.Module):
  '''Implementation of a Res UNet model.'''

  def __init__(self, in_channels, out_channels):
    '''
      ### Params:
         * `in_channels`: The number of channels in the input kernels (input shape [Batch, in_channels, H, W])
         * `out_channels`: The number of channels to output for each input. It's actually the number of classes for this problem.
    '''
    super(ResUNet, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels

    # Downsampling phase
    self.conv_down_1 = nn.Sequential(
      nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.Conv2d(64, 64, kernel_size=3, padding=1)
    )
    self.res_down_1 = self.res(in_channels, 64, 1)

    self.conv_down_2 = self.double_conv(64, 128, 2)
    self.res_down_2 = self.res(64, 128, 2)

    self.conv_down_3 = self.double_conv(128, 256, 2)
    self.res_down_3 = self.res(128, 256, 2)

    self.conv_down_4 = self.double_conv(256, 512, 2)
    self.res_down_4 = self.res(256, 512, 2)

    self.conv_down_5 = self.double_conv(512, 1024, 2)

    # Upsampling phase
    self.up_trans_1 = self.up_trans(1024, 512)
    self.conv_up_1 = self.double_conv(1024, 512)
    self.res_up_1 = self.res(1024, 512)

    self.up_trans_2 = self.up_trans(512, 256)
    self.conv_up_2 = self.double_conv(512, 256)
    self.res_up_2 = self.res(512, 256)

    self.up_trans_3 = self.up_trans(256, 128)
    self.conv_up_3 = self.double_conv(256, 128)
    self.res_up_3 = self.res(256, 128)

    self.up_trans_4 = self.up_trans(128, 64)
    self.conv_up_4 = self.double_conv(128, 64)
    self.res_up_4 = self.res(128, 64)

    # Output convolution
    self.out = nn.Conv2d(64, out_channels, kernel_size=1)

  def double_conv(self, in_channels, out_channels, stride=1):
    """Double convolution (each followed by a batch normalization and a ReLU)"""
    return nn.Sequential(
      # 1st convolution
      nn.BatchNorm2d(in_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
      # 2nd convolution
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
    )

  def res(self, in_channels, out_channels, stride=1):
    return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
      nn.BatchNorm2d(out_channels)
    )

  def up_trans(self, in_channels, out_channels):
    """Upsampling operation"""
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

  def forward(self, input):
    # Downsampling part (keep the outputs for the concatenation in the upsampling stage)
    down_conv_x1 = self.conv_down_1(input)
    down_res_x1 = self.res_down_1(input)
    down_x1 = torch.add(down_conv_x1, down_res_x1)

    down_conv_x2 = self.conv_down_2(down_x1)
    down_res_x2 = self.res_down_2(down_x1)
    down_x2 = torch.add(down_conv_x2, down_res_x2)

    down_conv_x3 = self.conv_down_3(down_x2)
    down_res_x3 = self.res_down_3(down_x2)
    down_x3 = torch.add(down_conv_x3, down_res_x3)

    down_conv_x4 = self.conv_down_4(down_x3)
    down_res_x4 = self.res_down_4(down_x3)
    down_x4 = torch.add(down_conv_x4, down_res_x4)

    down_x5 = self.conv_down_5(down_x4)

    # Upsampling part (concatenate outputs)
    up_x = self.up_trans_1(down_x5)
    conv_x = self.conv_up_1(torch.cat([down_x4, up_x], dim=1))
    res_x = self.res_up_1(torch.cat([down_x4, up_x], dim=1))

    up_x = self.up_trans_2(torch.add(conv_x, res_x))
    conv_x = self.conv_up_2(torch.cat([down_x3, up_x], dim=1))
    res_x = self.res_up_2(torch.cat([down_x3, up_x], dim=1))

    up_x = self.up_trans_3(torch.add(conv_x, res_x))
    conv_x = self.conv_up_3(torch.cat([down_x2, up_x], dim=1))
    res_x = self.res_up_3(torch.cat([down_x2, up_x], dim=1))

    up_x = self.up_trans_4(torch.add(conv_x, res_x))
    conv_x = self.conv_up_4(torch.cat([down_x1, up_x], dim=1))
    res_x = self.res_up_4(torch.cat([down_x1, up_x], dim=1))

    x = self.out(torch.add(conv_x, res_x))

    return x

  def pred(self, img, img_trans):
    '''
        Predict the `img` given the correct transformation for the img.
        If `img_trans` non then no transfromation is applied.
    '''
    self.eval()
    img = img_trans(img).unsqueeze(0)
    if torch.cuda.is_available():
      img = img.cuda()
    out = self.forward(img).cpu().detach()
    soft = nn.Softmax(dim=1)
    out_proba = soft(out)
    return (out_proba[:,1] > 0.5).float()