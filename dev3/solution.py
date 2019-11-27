import math
import pickle
import random

try:
    import cupy as np
except ModuleNotFoundError:
    import numpy as np

class NN(object):
    def __init__(self,
                 hidden_dims=(512, 256),
                 datapath='cifar10.pkl',
                 n_classes=10,
                 epsilon=1e-6,
                 lr=7e-4,
                 batch_size=1000,
                 seed=None,
                 activation="relu",
                 init_method="glorot"
                 ):

        self.hidden_dims = hidden_dims
        self.n_hidden = len(hidden_dims)
        self.datapath = datapath
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.init_method = init_method
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}

        if datapath is not None:
            u = pickle._Unpickler(open(datapath, 'rb'))
            u.encoding = 'latin1'
            self.train, self.valid, self.test = u.load()
        else:
            self.train, self.valid, self.test = None, None, None

    def initialize_weights(self, dims):
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = {}
        # self.weights is a dictionary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers

        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):
            bound = 1/math.sqrt(all_dims[layer_n-1])
            self.weights[f"W{layer_n}"] = np.random.uniform(-bound, bound, (all_dims[layer_n-1], all_dims[layer_n]))
            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))

    def relu(self, x, grad=False):
        isnt_arr = not (type(x) is np.array)
        if isnt_arr:
            x = np.array([x])
        else:
            x = x.copy()

        if grad:
            x[x>0] = 1
            x[x<=0] = 0
        else:
            np.maximum(x, 0, out=x)

        if isnt_arr:
            return x[0]
        else:
            return x

    def sigmoid(self, x, grad=False):
        if grad:
            return self.sigmoid(x) - self.sigmoid(x)**2

        return 1.0 / (1 + math.e ** (-x))

    def tanh(self, x, grad=False):
        if grad:
            return 1-self.tanh(x)**2

        isnt_arr = not (type(x) is np.array)
        if isnt_arr:
            x = np.array([x])

        x = np.tanh(x)
        if isnt_arr:
            return x[0]
        else:
            return x

    def activation(self, x, grad=False):
        if self.activation_str == "relu":
            func = self.relu
        elif self.activation_str == "sigmoid":
            func = self.sigmoid
        elif self.activation_str == "tanh":
            func = self.tanh
        else:
            raise Exception("invalid")

        return func(x, grad)

    def softmax(self, x):
        take_first = False
        if len(x.shape) == 1:
            take_first = True
            x = x.reshape((1, x.shape[0]))

        max = np.max(x, axis=1).reshape((x.shape[0], 1))
        x = np.exp(x - max)
        bot = np.sum(x, axis=1).reshape((x.shape[0], 1))
        ret = x / bot

        if take_first:
            return ret[0]
        else:
            return ret

    def forward(self, x):
        cache = {"Z0": x}
        # cache is a dictionary with keys Z0, A0, ..., Zm, Am where m - 1 is the number of hidden layers
        # Ai corresponds to the preactivation at layer i, Zi corresponds to the activation at layer i
        # WRITE CODE HERE

        for layer_n in range(1, self.n_hidden + 2):
            preactivation = np.matmul(cache[f"Z{layer_n-1}"], self.weights[f"W{layer_n}"]) + self.weights[f"b{layer_n}"]
            cache[f"A{layer_n}"] = preactivation
            if layer_n == self.n_hidden + 1:
                cache[f"Z{layer_n}"] = self.softmax(preactivation)
            else:
                cache[f"Z{layer_n}"] = self.activation(preactivation)

        return cache

    def backward(self, cache, labels):
        output = cache[f"Z{self.n_hidden + 1}"]
        grads = {}
        # grads is a dictionary with keys dAm, dWm, dbm, dZ(m-1), dA(m-1), ..., dW1, db1
        # WRITE CODE HERE
        pass
        return grads

    def update(self, grads):
        for layer_n in range(1, self.n_hidden + 2):
            self.weights[f"W{layer_n}"] -= self.lr * grads[f"dW{layer_n}"]
            self.weights[f"b{layer_n}"] -= self.lr * grads[f"db{layer_n}"]

    def one_hot(self, y):
        ret = np.zeros((len(y), self.n_classes))
        ret[np.arange(len(y)), y] = 1
        return ret

    def loss(self, prediction, labels):
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon
        # WRITE CODE HERE
        pass
        return 0

    def compute_loss_and_accuracy(self, X, y):
        one_y = self.one_hot(y)
        cache = self.forward(X)
        predictions = np.argmax(cache[f"Z{self.n_hidden + 1}"], axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(cache[f"Z{self.n_hidden + 1}"], one_y)
        return loss, accuracy, predictions

    def train_loop(self, n_epochs):
        X_train, y_train = self.train
        y_onehot = self.one_hot(y_train)
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]

                # MA PARTIE
                forward = self.forward(minibatchX)
                backward = self.backward(forward, minibatchY)
                self.update(backward)
                # FIN MA PARTIE

            X_train, y_train = self.train
            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
            X_valid, y_valid = self.valid
            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)

            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)

        return self.train_logs

    def evaluate(self):
        X_test, y_test = self.test
        loss, accuracy, predictions = self.compute_loss_and_accuracy(X_test, y_test)
        return loss, accuracy

#BONUS
def bonus_1():
    import matplotlib.pyplot as plt

    random.seed(0)
    nn = NN(lr=0.003, batch_size=100)

    epochs_n = 50

    nn.train_loop(epochs_n)

    epochs = list(np.arange(epochs_n))

    train_acc = nn.train_logs['train_accuracy']
    valid_acc = nn.train_logs['validation_accuracy']
    plt.plot(epochs ,train_acc , label="Train accuracy")
    plt.plot(epochs , valid_acc, label="Validation accuracy")
    plt.legend()
    plt.show()

    train_loss = nn.train_logs['train_loss']
    valid_loss = nn.train_logs['validation_loss']
    plt.plot(epochs , train_loss, label="Train loss")
    plt.plot(epochs , valid_loss, label="Validation loss")
    plt.legend()
    plt.show()