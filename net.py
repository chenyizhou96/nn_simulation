import torch.nn as nn
import torch 
import torch.nn.functional as F
import config

class Net(nn.Module):
  def __init__(self, input, output, config):
    super(Net, self).__init__()
    #self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    #self.conv1_drop = nn.Dropout2d()
    #self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    #self.conv3 = nn.Conv2d(20,20,kernel_size = 5)
    #self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(input, config.layer_size)
    self.fc2 = nn.Linear(config.layer_size, config.layer_size)
    self.fc3 = nn.Linear(config.layer_size, config.layer_size)
    self.fc4 = nn.Linear(config.layer_size, output)
    self.activation = F.relu
    if config.activation == 1:
      self.activation = torch.sigmoid
    elif config.activation == 2:
      self.activation = torch.tanh



  def forward(self, x):
    x = self.activation(self.fc1(x))
    x = self.activation(self.fc2(x))
    x = self.activation(self.fc3(x))
    return self.activation(self.fc4(x))