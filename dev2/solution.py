import numpy as np

def coerce(vector):
    return np.reshape(vector, (len(vector), 1))

class SVM:
    def __init__(self,eta, C, niter, batch_size, verbose):
        self.eta = eta; self.C = C; self.niter = niter; self.batch_size = batch_size; self.verbose = verbose

    def make_one_versus_all_labels(self, y, m):
        fill = np.full((len(y), m), -1)
        for i, label in enumerate(y):
            fill[i][label] = 1
        return fill

    # For a w_v weight vector, a features matrix, and the indicating function appropriate for w_v for features
    # ex self.loss(self.w.T[0].T, x, y[:,0])
    def loss(self, w_v, features, labels):
        feature_weights = features.dot(w_v)
        label_tested = coerce(feature_weights * labels)   # why le reshape necessaire?

        maxed = np.max(1 - label_tested, axis=1, initial=0)

        return coerce(maxed)

    def for_wv_in_w(self):
        for i, wi in enumerate(self.w.T):
            yield i, wi.T

    def compute_loss(self, x, y):
        """
        x : numpy array of shape (minibatch size, 401) FEATURES
        y : numpy array of shape (minibatch size, 10) LABELS in -1,1
        returns : float
        """
        sums = 0
        for i, wi in self.for_wv_in_w():
            loss = self.loss(wi, x, y[:, i])
            pwr = np.power(loss, 2)
            sums += np.sum(pwr)
        sums *= self.C / len(x)

        w_norm = np.linalg.norm(self.w, axis=0)
        w_pwr = np.power(w_norm, 2)
        weight_stuff = np.sum(w_pwr) / 2

        return weight_stuff + sums

    def compute_gradient(self, x, y):
        """
        x : numpy array of shape (minibatch size, 401)
        y : numpy array of shape (minibatch size, 10)
        returns : numpy array of shape (401, 10)
        """

        gradient = np.zeros(self.w.shape).T

        for i, wi in self.for_wv_in_w():
            labels = y.T[i]

            loss = self.loss(wi, x, labels)

            per_feat = loss * x
            tested = coerce(labels) * per_feat
            pre_gradient = np.sum(tested, axis=0)
            gradient[i] = ((-2*self.C)/len(x)) * pre_gradient
            gradient[i] += wi   #

        return gradient.T

    # Batcher function
    def minibatch(self, iterable1, iterable2, size=1):
        l = len(iterable1)
        n = size
        for ndx in range(0, l, n):
            index2 = min(ndx + n, l)
            yield iterable1[ndx: index2], iterable2[ndx: index2]

    def infer(self, x):
        """
        x : numpy array of shape (number of examples to infer, 401)
        returns : numpy array of shape (number of examples to infer, 10)
        """
        num_classes = self.w.shape[1]
        num_data = len(x)

        scores = np.zeros((num_classes, num_data))
        for i, wi in self.for_wv_in_w():
            scores[i] = x.dot(wi)

        preds = np.argmax(scores, axis=0)
        return self.make_one_versus_all_labels(preds, num_classes)

    def compute_accuracy(self, y_inferred, y):
        """
        y_inferred : numpy array of shape (number of examples, 10)
        y : numpy array of shape (number of examples, 10)
        returns : float
        """
        equals = (y_inferred == y)
        all = np.all(equals, axis=1)
        return np.mean(all)

    def fit(self, x_train, y_train, x_test, y_test):
        """
        x_train : numpy array of shape (number of training examples, 401)
        y_train : numpy array of shape (number of training examples, 10)
        x_test : numpy array of shape (number of training examples, 401)
        y_test : numpy array of shape (number of training examples, 10)
        returns : float, float, float, float
        """
        self.num_features = x_train.shape[1]
        self.m = y_train.max() + 1
        y_train = self.make_one_versus_all_labels(y_train, self.m)
        y_test = self.make_one_versus_all_labels(y_test, self.m)
        self.w = np.zeros([self.num_features, self.m])

        for iteration in range(self.niter):
            # Train one pass through the training set
            for x, y in self.minibatch(x_train, y_train, size=self.batch_size):
                grad = self.compute_gradient(x,y)
                self.w -= self.eta * grad

            # Measure loss and accuracy on training set
            train_loss = self.compute_loss(x_train,y_train)
            y_inferred = self.infer(x_train)
            train_accuracy = self.compute_accuracy(y_inferred, y_train)

            # Measure loss and accuracy on test set
            test_loss = self.compute_loss(x_test,y_test)
            y_inferred = self.infer(x_test)
            test_accuracy = self.compute_accuracy(y_inferred, y_test)

            if __name__ == "__main__":
                saveit(self.C, train_loss, train_accuracy, test_loss, test_accuracy)

            if self.verbose:
                print("Iteration %d:" % iteration)
                print("Train accuracy: %f" % train_accuracy)
                print("Train loss: %f" % train_loss)
                print("Test accuracy: %f" % test_accuracy)
                print("Test loss: %f" % test_loss)
                print("")

        return train_loss, train_accuracy, test_loss, test_accuracy

attr_list = ["Train loss", "Train accuracy", "Test loss", "Test accuracy"]
c_list = [0.1, 1, 30, 50]
dico = {}
for c in c_list:
    temp = {}
    for attr in attr_list:
        temp[attr] = []
    dico[c] = temp

def saveit(c, train_loss, train_acc, test_loss, test_acc):
    global dico
    temp = dico[c]
    temp["Train loss"].append(train_loss)
    temp["Train accuracy"].append(train_acc)
    temp["Test loss"].append(test_loss)
    temp["Test accuracy"].append(test_acc)

    print(dico)

def plotit():
    global dico
    print(dico[1][attr_list[0]])
    x_axis = list(range(len(dico[1][attr_list[0]])))

    import matplotlib.pyplot as plt

    for attr in attr_list:
        attr_has_loss = "loss" in attr

        for c in [0.1, 1, 30, 50]:
            if c == 50 and attr_has_loss:
                continue

            plt.plot(x_axis, dico[c][attr], label="C value: "+str(c))

        if attr_has_loss:
            plt.ylabel(attr+" (%)")
        else:
            plt.ylabel(attr)

        plt.legend(loc='upper left')
        plt.xlabel("Epochs")
        plt.title(attr)
        plt.show()

if __name__ == "__main__":

    # Load the data files
    print("Loading data...")
    x_train = np.load("train_features.npy")
    x_test = np.load("test_features.npy")
    y_train = np.load("train_labels.npy")
    y_test = np.load("test_labels.npy")

    for c in c_list:
        svm = SVM(eta=0.001, C=c, niter=200, batch_size=5000, verbose=True)
        print("Fitting the model...")
        train_loss, train_accuracy, test_loss, test_accuracy = svm.fit(x_train, y_train, x_test, y_test)

    # # to infer after training, do the following:
    #y_inferred = svm.infer(x_test)

    ## to compute the gradient or loss before training, do the following:
    y_train_ova = svm.make_one_versus_all_labels(y_train, 10) # one-versus-all labels
    svm.w = np.zeros([401, 10])
    grad = svm.compute_gradient(x_train, y_train_ova)
    loss = svm.compute_loss(x_train, y_train_ova)
    print(loss)
    print(grad)

    plotit()
