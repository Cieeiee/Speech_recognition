import urllib.request
import numpy as np
import os


def _download(file_path):
    file_name = file_path.split('/')[-1]
    urllib.request.urlretrieve(
        file_path,
        file_name,
        lambda block_num, block_size, total_size:
        print(f'\r Downloading... {min(block_size * block_num / total_size * 100, 100):.1f}%', end='')
    )


test = np.zeros(5)
print(test)
print(test.shape)
