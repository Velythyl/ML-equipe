import math

import numpy as np

x = np.array([-1, 1])
print(np.where(x>0))

n_classes = 5
def one_hot(y):
    ret = np.zeros((len(y), n_classes))
    ret[np.arange(len(y)),y] = 1
    return ret

print(one_hot(np.array([4,1,2,3,4])))


def relu(x, grad=False):
    isnt_arr = not (type(x) is np.array)
    if isnt_arr:
        x = np.array([x])
    else:
        x = x.copy()

    if grad:
        x[x > 0] = 1
        x[x <= 0] = 0
    else:
        np.maximum(x, 0, out=x)

    if isnt_arr:
        return x[0]
    else:
        return x

print(relu(np.array([[3,4,3,-1],[-1,3,4,5]]), grad=True))
print(relu(2))
print(relu(-1))


def sigmoid(x, grad=False):
    if grad:
        return sigmoid(x) - sigmoid(x) ** 2

    return 1.0 / (1 + math.e ** (-x))

print(sigmoid(np.array([[3,4,3,-1],[-1,3,4,5]])))
print(sigmoid(2))
print(sigmoid(-1))


def tanh(x, grad=False):
    if grad:
        return 1 - tanh(x) ** 2

    isnt_arr = not (type(x) is np.array)
    if isnt_arr:
        x = np.array([x])

    x = np.tanh(x)
    if isnt_arr:
        return x[0]
    else:
        return x

print(tanh(np.array([[3,4,3,-1],[-1,3,4,5]]), grad=True))
print(tanh(2))
print(tanh(-1))


def softmax(x):
    max = np.max(x, axis=1).reshape((x.shape[0], 1))
    x = np.exp(x - max)
    bot = np.sum(x, axis=1).reshape((x.shape[0], 1))
    return x / bot

print(softmax(np.array([[3,4,3,-1],[-1,3,4,5]])))
print(softmax(2))
print(softmax(-1))