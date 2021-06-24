import torch
import torch.nn as nn
import torchvision
from torch import optim
import numpy as np

# input: [batch_size, in_channels, features]
class Late_Merge_Siamese(nn.Module):
  def __init__(self, hyperparam, activ_bool = True):
    super(Late_Merge_Siamese, self).__init__()

    self.activ_func = nn.ReLU(inplace = activ_bool)

    self.cnn_1 = nn.Sequential(
      nn.Conv1d(1, hyperparam['cnn_1'], kernel_size = 3),
      self.activ_func,
      #nn.MaxPool1d(hyperparam['max_pool_1'])
    )

    self.cnn_2 = nn.Sequential(
      nn.Conv1d(hyperparam['cnn_1'], hyperparam['cnn_2'], kernel_size = 3),
      self.activ_func,
      nn.BatchNorm1d(num_features = 64),
      nn.Dropout(0.2),
      nn.MaxPool1d(hyperparam['max_pool_2'])
    )

    # Add avg adaptive pooling and see
    self.linear = nn.Sequential(
      nn.Linear(14*hyperparam['cnn_2'], hyperparam['lin_1']),
      self.activ_func,
      nn.Linear(hyperparam['lin_1'], hyperparam['lin_2']),
      self.activ_func,
      nn.Dropout(0.2),
      nn.Linear(hyperparam['lin_2'], hyperparam['lin_3']),
    )

  def forward_once(self, x):
    x1 = self.cnn_1(x)
    x2 = self.cnn_2(x1)
    #print(x2.shape)

    x2_out = x2.view(x2.size()[0], -1)
    out = self.linear(x2_out)

    return out

  def forward(self, left, right):
    out2 = self.forward_once(right)
    out1 = self.forward_once(left)

    return out1, out2