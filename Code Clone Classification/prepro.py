"""Get Data From Csv File"""

import pandas as pd
import numpy as np

# Dataset Length: 300201 
def get_csv_data(csv_path, feature_len = 32):
  csv_data = pd.read_csv(csv_path, sep=',', header=None)
  data = csv_data.values.astype(np.float)[:, 0:2*feature_len]
  labels = csv_data.values.astype(np.float)[:,2*feature_len:]

  total_data = np.hstack((data, labels))

  np.random.shuffle(total_data)

  shuffled_data = total_data[:, :-1]
  shuffled_labels = total_data[:, -1]

  left_data = shuffled_data[:, 0:feature_len]
  right_data = shuffled_data[:, feature_len: 2*feature_len]
  
  return left_data, right_data, shuffled_labels