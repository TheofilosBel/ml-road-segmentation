import torch.nn as nn

class FCN(nn.Module):
  def __init__(self, in_channels, out_channels, linear_len):
    '''
        ### Params:
         * `in_channels`: The number of channels in the input kernels (input shape [Batch, in_channels, H, W])
         * `out_channels`: The number of channels to output for each input. It's actually the number of classes for this problem.
         * `linear_len`: The input size that the fully connected layer should have. Its the size of the flattened output of the CNN layers.
    '''
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
