from load_data import load_data
import numpy as np

# load_data()
data = np.load('train_data.npy')
label = np.load('train_label.npy')
print(data.shape, label.shape)
