from train import train
from prepro import get_csv_data

"""Main"""

if __name__ == '__main__':
  csv_path = '/content/drive/MyDrive/Code Clone Detection/syntax_semantic.csv'
  left_numpy, right_numpy, labels = get_csv_data(csv_path)

CUDA_LAUNCH_BLOCKING=1

"""
Late Merge Siamese : 1
No Siamese : 2
Intermediate Merge Siamese: 3

"""

train(left_numpy, right_numpy, labels, 1)




"""EXTRA"""

def extra(labels):
  ones, zeros = 0,0
  for i in labels:
    if int(i[0]) == 1:
      ones+=1
    else:
      zeros+=1
  return ones, zeros
one, zero = extra(labels)
print('ones: ', one)
print('zeros: ', zero)