import torch.nn as nn

class FCN(nn.Module):
  def __init__(self, in_channels, out_channels, linear_len):
    super(FCN, self).__init__()

    self.cnn_layers = nn.Sequential(        
      # Defining a 2D convolution layer
      nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),

      # Defining another 2D convolution layer
      nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )

    self.linear_layers = nn.Sequential(
      nn.Linear(linear_len, 512),
      nn.ReLU(inplace=True),
      nn.Linear(512, out_channels)        
    )

  # Defining the forward pass    
  def forward(self, x):
    x = self.cnn_layers(x)
    x = x.view(x.size(0), -1)    
    x = self.linear_layers(x)
    return x
    