"""Training the model"""

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import torch.utils.data as Data
from models.IMSiamese import Inter_Merge_Siamese
from models.LMSiamese import Late_Merge_Siamese
from models.NoSiamese import NoSiamese
from dataloader import Dataloader
from contrastiveloss import ContrastiveLoss
from utilities import count, threshold_contrastive
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

"""
To view the data written by tensorboardX
tensorboard --logdir <path of logs directory>
In my case, pathdir = 'logs/'
"""

#os.makedirs('/drive/MyDrive', SAVE_DIR, exist_ok=True)

hyperparam = {
    'cnn_1': 32,
    'cnn_2': 64,
    'max_pool_1': 2,
    'max_pool_2': 2,
    'lin_1': 512,
    'lin_2': 256,
    'lin_3': 128,      
  }


def init_weights(model):
  for name, param in model.named_parameters():
    nn.init.uniform_(param.data, -0.08, 0.08)

def train(left, right, labels, arch, contra_loss = True):
  data = Dataloader(left, right, labels)
  logger = SummaryWriter(os.path.join(HOME, LOG_DIR, TIME + ': Code Clone'))
  dataset_len = len(labels)

  opt = {
      'batch_sz': 100,
      'lr': 0.0001,
      'epochs': 30,
      'momentum': 0.09,
      'train_len': int(0.70*dataset_len),
      'val_len': int(0.85*dataset_len),
      'test_len': int(dataset_len),
  }

  architecture_dict = {
    1: Late_Merge_Siamese(hyperparam),
    2: NoSiamese(hyperparam),
    3: Inter_Merge_Siamese(hyperparam)
}

  train_loss_arr = []
  val_loss_arr = []

  feature_len = 32
  model = architecture_dict[arch]
  #model = Inter_Merge_Siamese(hyperparam)
  #model.apply(init_weights)

  train_loader = Data.DataLoader(Data.Subset(data, range(opt['train_len'])), batch_size = opt['batch_sz'], shuffle = True)
  val_loader = Data.DataLoader(Data.Subset(data, range(opt['train_len'], opt['val_len'])), batch_size = opt['batch_sz'] ,shuffle = True)
  test_loader = Data.DataLoader(Data.Subset(data, range(opt['val_len'], opt['test_len'])), batch_size = opt['batch_sz'],shuffle = True)

  optimizer = optim.Adam(model.parameters(), lr = opt['lr'])
  #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
  
  if arch == 1:
    loss_fn = ContrastiveLoss()
  else:
    loss_fn = torch.nn.CrossEntropyLoss()
  
  #loss_fn = nn.CrossEntropyLoss()
  model.cuda()

  print('-----------------BEGIN TRAINING-------------------')

  for epoch in range(opt['epochs']):
    train_loss = 0.0
    model.train()
    a = list(model.parameters())[0].clone()
    train_len = 0.0
    val_len = 0.0

    for left_vec, right_vec, label in train_loader:
      
      left_vec, right_vec= torch.unsqueeze(left_vec, 1), torch.unsqueeze(right_vec, 1)
      left_vec = left_vec.to(device)
      right_vec = right_vec.to(device)
      label = label.to(device)
      
      optimizer.zero_grad()
      
      if arch == 1: # LATE_MERGE_SIAMESE
        out1, out2 = model(left_vec, right_vec)
        loss = loss_fn(out1, out2, label)
      elif arch == 2: # NO_SIAMESE
        cat_vec = torch.cat((left_vec, right_vec),2)
        out = model(cat_vec)
        loss = loss_fn(out, label)
      else: # INTERMEDIATE_MERGE_SIAMESE
        out = model(left_vec, right_vec)
        loss = loss_fn(out, label)
      
      #out = model(left_vec, right_vec)
      #loss = loss_fn(out,label)
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      train_len += label.shape[0]

    b = list(model.parameters())[0].clone()
    compare = torch.equal(a.data,b.data)
    print('BOOL: ', compare)
    logger.add_scalar('Training_loss', train_loss/train_len, epoch+1)
    print()
    print('EPOCH: ', epoch, '---', train_loss/train_len)
    train_loss_arr.append(train_loss/train_len)

    print('---------------------------BEGIN VALIDATION---------------------------')
    
    val_loss = 0.0
    val_acc = 0.0
    
    temp = True
    model.eval()
    with torch.no_grad():
      for left_vec, right_vec, label in val_loader:
        left_vec, right_vec = torch.unsqueeze(left_vec, 1), torch.unsqueeze(right_vec, 1)
        left_vec = left_vec.to(device)
        right_vec = right_vec.to(device)
        label = label.to(device)
        
        if arch == 1: # LATE_MERGE_LOSS
          out1, out2 = model(left_vec, right_vec)
          loss = loss_fn(out1, out2, label)

          if contra_loss:
            output_labels = threshold_contrastive(out1, out2)
          else:
            eucledian_distance = F.pairwise_distance(out1, out2)
            output_labels = torch.sigmoid(eucledian_distance)

        elif arch == 2: # NO_SIAMESE
          cat_vec = torch.cat((left_vec, right_vec),2)
          out = model(cat_vec)
          output_labels = torch.max(out, 1)[1]

        else: # INTERMEDIATE_MERGE_SIAMESE
          out = model(left_vec, right_vec)
          loss = loss_fn(out, label)
          output_labels = torch.max(out, 1)[1]
        
        #out = model(left_vec, right_vec)
        #loss = loss_fn(out, label)
        #output_labels = torch.max(out, 1)[1]
        label = torch.squeeze(label)
        output_labels = torch.squeeze(output_labels)
        pred = output_labels.data.cpu().numpy()
        target = label.data.cpu().numpy()

        if temp:
          #print('OUT: ', output_labels)
          print('OUT2: ', output_labels.shape)
          print('OUT3: ', label.shape)
          #print('OUTPUT: ', out.shape)
          print('OUT_ONES: ', count(output_labels))
          print('OUT_LABELS: ', count(label))
          print('TORCH: ', float((pred == target).sum()))
          temp = False
  
        old_val_acc = val_acc
        val_len += label.shape[0]
        val_acc += float((pred == target).sum())
        
        val_loss += loss.item()

    #print('VAL:', val_loss)
    print(f'Epoch {epoch+0:03}: | Train Loss: {train_loss/train_len:.5f} | Val Loss: {val_loss/val_len:.5f} | Val Acc: {val_acc/val_len:.3f}')
    torch.cuda.empty_cache()
    val_loss_arr.append(val_loss/val_len)

  plt.figure(figsize=(10,5))
  plt.title("Training and Validation Loss")
  plt.plot(val_loss_arr,label="val")
  plt.plot(train_loss_arr,label="train")
  plt.xlabel("iterations")
  plt.ylabel("Loss")
  plt.legend()
  plt.show()
  logger.close()
  torch.save({'state_dict': model.state_dict()}, os.path.join(HOME, 'fusional_snn.pt'))
  print('TRAINING DONE')
  test(model, device, test_loader, arch, contra_loss)
  
      
if __name__ == '__main__':
   LOG_DIR = 'logs'
   HOME = '/drive/Mydrive'
   SAVE_DIR = 'save'
   TIME = time.strftime("%Y%m%d_%H%M%S")
   device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def test(model, device, test_loader, arch, contra_loss):
  model.eval().to(device)
  y = {'Actual': [], 'Predicted': []}
  with torch.no_grad():
    for left_vec, right_vec, label in test_loader:
      left_vec, right_vec = torch.unsqueeze(left_vec, 1), torch.unsqueeze(right_vec, 1)
      left_vec = left_vec.to(device)
      right_vec = right_vec.to(device)
      label = label.to(device)
      
      if arch == 1:
        out1, out2 = model(left_vec, right_vec)
        if contra_loss:
          output_labels = threshold_contrastive(out1, out2)
        else:
          eucledian_distance = F.pairwise_distance(out1, out2)
          output_labels = torch.sigmoid(eucledian_distance)
      elif arch == 2:
        cat_vec = torch.cat((left_vec, right_vec), 2)
        out = model(cat_vec)
        output_labels = torch.max(out, 1)[1]
      else:
        out = model(left_vec, right_vec)
        output_labels = torch.max(out, 1)[1]
      
      #out = model(left_vec, right_vec)
      #output_labels = torch.max(out, 1)[1]
      label = torch.squeeze(label)
      output_labels = torch.squeeze(output_labels)
      pred = output_labels.data.cpu().numpy()
      target = label.data.cpu().numpy()

      y['Actual'].extend(target.tolist())
      y['Predicted'].extend(pred.tolist())

  print('\n f1 Score= %.4f' % f1_score(y['Actual'], y['Predicted']))
  print('Precision= %.4f' % precision_score(y['Actual'], y['Predicted'], zero_division=0))
  print(' Recall= %.4f' % recall_score(y['Actual'], y['Predicted'])) 
 
  print('\nAccuracy: %.4f' % accuracy_score(y['Actual'], y['Predicted']))
