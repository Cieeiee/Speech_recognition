from pathlib import Path
import numpy as np
import librosa
import tensorflow as tf

def load_data():
    data_paths = [str(x) for x in Path('.').glob('../data/*.wav')]
    for path in data_paths:
      y, sr = librosa.load(path)
      feature = librosa.feature.mfcc(y=y, sr=sr)
      

def load_label():
    text_paths = list(Path('.').glob('../data/*.trn'))
    labels = []
    for path in text_paths:
        with path.open(encoding='utf8') as f:
            text = f.readline()
            labels.append(text.split())
    chars = []
    for label in labels:
        for text in label:
            if text not in chars:
                chars.append(text)
    char2id = {c: i for i, c in enumerate(chars)}
    id2char = {i: c for i, c in enumerate(chars)}
    return char2id, id2char, labels
