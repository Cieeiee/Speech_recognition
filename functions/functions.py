import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def identity_function(x):
    return x


def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y