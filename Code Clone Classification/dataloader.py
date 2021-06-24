"""Dataloader for the Dataset"""

import torch
import torchvision
import torch.utils.data as Data

class Dataloader(Data.Dataset):
  def __init__(self, left_arr, right_arr, labels):
    super(Dataloader).__init__()

    self.left_tensor = torch.from_numpy(left_arr).float()
    self.right_tensor = torch.from_numpy(right_arr).float()
    self.label = torch.from_numpy(labels).long()
    self.len = len(labels)


  def __len__(self):
    return self.len
  
  def __getitem__(self, idx):
    return (self.left_tensor[idx], self.right_tensor[idx], self.label[idx])
