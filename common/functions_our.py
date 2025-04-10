from common.np import *


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    if x.ndim == 2:
        # with batch
        x = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
        return x / np.sum(x, axis=1).reshape(-1, 1)
    elif x.ndim == 1:
        x = np.exp(x - np.max(x))
        return x / np.sum(x)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    # np.arange(y.shape[0]) -> [0, 1, 2, ..., y.shape[0] - 1]
    # t -> [2, 7, 0, 9, ...]
    return -np.sum(np.log(y[np.arange(y.shape[0]), t] + 1e-7)) / y.shape[0]
