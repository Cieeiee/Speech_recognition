from pathlib import Path
import numpy as np
import librosa


def load_data():
    return


def load_label():
    text_paths = list(Path('.').glob('../data/*.trn'))
    labels = []
    paths = []
    for path in text_paths:
        with path.open(encoding='utf8') as f:
            text = f.readline()
            labels.append(text.split())
            paths.append(path.parent / path.name.rstrip('.trn'))
    chars = []
    for label in labels:
        for text in label:
            if text not in chars:
                chars.append(text)
    char2id = {c: i for i, c in enumerate(chars)}
    id2char = {i: c for i, c in enumerate(chars)}
    return char2id, id2char, labels, paths


def batch_generator(label, data, char2id, batch_size=16):
    offset = 0
    while True:
        if offset + batch_size > len(label) or offset == 0:
            data_index = np.arange(len(label))
            np.random.shuffle(data_index)
            label = [label[i] for i in data_index]
            data = [data[i] for i in data_index]
            offset = 0
        data_batch = data[offset: offset + batch_size]
        label_batch = label[offset: offset + batch_size]
        data_max = max([data_batch[i].shape[0] for i in range(batch_size)])
        label_max = max([len(label_batch[i]) for i in range(batch_size)])
        batch_data = np.zeros((batch_size, data_max, 13))
        batch_label = np.ones((batch_size, label_max))
        data_length = np.zeros((batch_size, 1), dtype='int32')
        label_length = np.zeros((batch_size, 1), dtype='int32')
        for i in range(batch_size):
            data_length[i, 0] = data_batch[i].shape[0]
            batch_data[i, :data_length[i, 0]] = data_batch[i]
            label_length[i, 0] = len(label_batch[i])
            batch_label[i, :label_length[i, 0]] = np.array([char2id[c] for c in label_batch[i]])
        inputs = {
            'data': batch_data,
            'label': batch_label,
            'data_length': data_length,
            'label_length': label_length
        }
        outputs = {'ctc': np.zeros(batch_size)}
        yield inputs, outputs


