import cv2
import pandas
import platform
import numpy as np
from pathlib import Path

def load_data():
  data_paths = [str(x) for x in Path('.').glob('data/Train/*')]
  native_label = pandas.read_csv('data/Train_label.csv')
  images = []
  label = []
  for index, path in enumerate(data_paths):
    image = cv2.imread(path)
    if image.shape[0] > image.shape[1]:
      image = image[:image.shape[1]]
    image = cv2.resize(image, (244, 244))
    images.append(image)
    if platform.system() == 'Windows':
      file_name = path.split('\\')[-1]
    else:
      file_name = path.split('/')[-1]
    if index % 500 == 0: print(index)
    label.append(native_label.loc[native_label['FileName'] == file_name, 'Code'].values[0])
  images = np.array(images)
  label = np.array(label)
  np.save('train_data.npy', images)
  np.save('train_label.npy', label)