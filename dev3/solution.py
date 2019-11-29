import math
import pickle
import random

try:
    #!pip install cupy-cuda101
    import cupy as np
except ModuleNotFoundError:
    print("Not using cupy")
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
                 init_method="glorot",
                 verbose=False
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
        self.verbose = verbose

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}

        if datapath is not None:
            u = pickle._Unpickler(open(datapath, 'rb'))
            u.encoding = 'latin1'
            self.train, self.valid, self.test = u.load()

            self.train = (np.asarray(self.train[0]), np.asarray(self.train[1]))
            self.valid = (np.asarray(self.valid[0]), np.asarray(self.valid[1]))
            self.test = (np.asarray(self.test[0]), np.asarray(self.test[1]))
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

    def is_numeirc(self, x):
        try:
            temp = x.shape
            return False
        except:
            return True

    def relu(self, x, grad=False):
        isnt_arr = self.is_numeirc(x)

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

        isnt_arr = self.is_numeirc(x)
        if isnt_arr:
            x = np.array([x])

        #x = np.tanh(x)

        e2x = np.exp(2*x)
        x = (e2x - 1) / (e2x + 1)

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

        for layer_n in reversed(range(1, self.n_hidden + 2)):
            if layer_n == self.n_hidden + 1:
                dA = cache[f"Z{layer_n}"] - labels  # dL/oa
            else:
                dA = self.activation(cache[f"Z{layer_n}"], grad=True) * grads[f"dZ{layer_n+1}"]
                grads[f"dA{layer_n}"] = dA

            dW = (dA.T @ cache[f"Z{layer_n-1}"]).T / len(labels)    # d oa/dW2 * dL/oa
            db = np.mean(dA, axis=0, keepdims=True)

            dZ = dA @ self.weights[f"W{layer_n}"].T


            grads[f"dW{layer_n}"] = dW
            grads[f"db{layer_n}"] = db
            grads[f"dZ{layer_n}"] = dZ

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

        """
        reduced = prediction[np.argmax(labels,axis=1).reshape(-1,1)]
        reduced = np.mean(np.log(reduced))
        return -reduced"""

        t = np.log(prediction) * labels
        t = np.sum(t, axis=1)
        t = np.mean(t)

        return -t

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

            if self.verbose:
                print(epoch+1, "of", n_epochs, "done!")

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
    import time

    nn = NN(lr=0.003, batch_size=100, verbose=True, seed=0, activation="relu", hidden_dims=(512,256), datapath="drive/My Drive/cifar10.pkl")


    epochs_n = 50

    start = time.time()
    nn.train_loop(epochs_n)
    print("Took",time.time()-start, "seconds to train")

    epochs = list(np.arange(epochs_n))

    train_acc = nn.train_logs['train_accuracy']
    print(train_acc)
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


def bonus_3_1():
    def iter(x):
        return 1636146 + (x-1) * 14520

    x = 0
    goal = 1707274
    diff = goal - iter(0)
    best_x = 0
    while True:
        t = abs(iter(x) - goal)
        if t < diff:
            best_x = x
            diff = t
        if t > diff:
            print(best_x)
            print(iter(best_x))
            break
        x += 1

def bonus_3_2():
    import matplotlib.pyplot as plt
    import time

    nn = NN(lr=0.003, batch_size=100, verbose=True, seed=0, activation="relu", hidden_dims=(512,120,120,120,120,120,120), datapath="drive/My Drive/cifar10.pkl")


    epochs_n = 50

    start = time.time()
    nn.train_loop(epochs_n)
    print("Took",time.time()-start, "seconds to train")

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

#bonus_3_2()

def bonus_4():
    import matplotlib.pyplot as plt
    import time

    epochs_n = 50
    epochs = [i for i in range(epochs_n)]

    logs = [[],[]]
    for i in range(1, 4):
        nn1 = NN(lr=0.003, batch_size=100, verbose=True, seed=i, activation="relu",
                 hidden_dims=(512,256), datapath="drive/My Drive/cifar10.pkl")

        nn2 = NN(lr=0.003, batch_size=100, verbose=True, seed=i, activation="relu", hidden_dims=(512,120,120,120,120,120,120), datapath="drive/My Drive/cifar10.pkl")

        nn1.train_loop(epochs_n)
        nn2.train_loop(epochs_n)

        logs[0].append(nn1.train_logs)
        logs[1].append(nn2.train_logs)

    vocab = ["train_accuracy", "validation_accuracy"]
    dico = {}
    for word in vocab:
        for j in [0,1]:
            for i in range(len(logs[0])):
                item = logs[j][i][word]
                key = word+"_NN"+str(j+1)
                if key in dico:
                    dico[key].append(item)
                else:
                    dico[key] = [item]

    for key in dico.keys():
        try:
            arr = np.array(dico[key])
        except Exception:
            for l in dico[key]:
                for i, ele in enumerate(l):
                    l[i] = float(ele)
            arr = np.array(dico[key])

        mean = list(np.mean(arr, axis=0))
        std = list(10 * np.std(arr, axis=0))

        plt.errorbar(epochs, mean, std, label=key.replace("_", " "))

    plt.legend()
    plt.show()

#bonus_4()