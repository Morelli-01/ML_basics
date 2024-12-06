import numpy as np
import sklearn.svm
from numpy.linalg import norm

from datasets import gaussians_dataset, people_dataset, two_moon_dataset
from utils import plot_pegasos_margin
from sklearn.svm import SVC

eps = np.finfo(float).eps


class SVM:
    """ Models a Support Vector machine classifier based on the PEGASOS algorithm. """

    def __init__(self, n_epochs, lambDa, use_bias=True):
        """ Constructor method """

        # weights placeholder
        self._w = None
        self._original_labels = None
        self._n_epochs = n_epochs
        self._lambda = lambDa
        self._use_bias = use_bias

    def map_y_to_minus_one_plus_one(self, y):
        """
        Map binary class labels y to -1 and 1
        """
        ynew = np.array(y)
        self._original_labels = np.unique(ynew)
        assert len(self._original_labels) == 2
        ynew[ynew == self._original_labels[0]] = -1.0
        ynew[ynew == self._original_labels[1]] = 1.0
        return ynew

    def map_y_to_original_values(self, y):
        """
        Map binary class labels, in terms of -1 and 1, to the original label set.
        """
        ynew = np.array(y)
        ynew[ynew == -1.0] = self._original_labels[0]
        ynew[ynew == 1.0] = self._original_labels[1]
        return ynew

    def loss(self, y_true, y_pred):
        """
        The PEGASOS loss term

        Parameters
        ----------
        y_true: np.array
            real labels in {0, 1}. shape=(n_examples,)
        y_pred: np.array
            predicted labels in [0, 1]. shape=(n_examples,)

        Returns
        -------
        float
            the value of the pegasos loss.
        """

        """
        Write HERE the code for computing the Pegasos loss function.
        """
        loss = 0
        for i in range(y_true.shape[0]):
            tmp = 1 - y_true[i] * y_pred[i]
            if tmp > 0:
                loss += tmp

        loss = self._lambda / 2 * np.linalg.norm(self._w) ** 2 + 1 / y_true.shape[0] * loss

        return loss

    def fit_gd(self, X, Y, verbose=False):
        """
        Implements the gradient descent training procedure.

        Parameters
        ----------
        X: np.array
            data. shape=(n_examples, n_features)
        Y: np.array
            labels. shape=(n_examples,)
        verbose: bool
            whether or not to print the value of cost function.
        """

        if self._use_bias:
            X = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=-1)

        n_samples, n_features = X.shape
        Y = self.map_y_to_minus_one_plus_one(Y)

        # initialize weights
        self._w = np.zeros(shape=(n_features,), dtype=X.dtype)

        t = 0
        grad = 0
        # loop over epochs
        for e in range(1, self._n_epochs + 1):
            # plot_pegasos_margin(X, Y, self)

            for j in range(n_samples):
                """
                Write HERE the update step.
                """
                if Y[j] * self._w @ X[j] < 1:
                    grad += Y[j] * X[j]

            grad = self._lambda * self._w - 1 / n_samples * grad  # predict training data
            eta = 1 / (e * self._lambda)
            self._w -= eta * grad
            cur_prediction = np.dot(X, self._w)

            # compute (and print) cost
            cur_loss = self.loss(y_true=Y, y_pred=cur_prediction)

            if verbose:
                print("Epoch {} Loss {}".format(e, cur_loss))

    def predict(self, X):

        if self._use_bias:
            X = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=-1)

        """
        Write HERE the criterium used during inference. 
        W * X > 0 -> positive class
        X * X < 0 -> negative class
        """
        predictions = X @ self._w
        # return np.where(np.random.choice(2, X.shape[0]) > 0.0,
        #                 self._original_labels[1], self._original_labels[0])
        return predictions > 0

    def fit_gd_pegasus(self, X, Y, verbose=False):
        """
        Implements the gradient descent training procedure.

        Parameters
        ----------
        X: np.array
            data. shape=(n_examples, n_features)
        Y: np.array
            labels. shape=(n_examples,)
        verbose: bool
            whether or not to print the value of cost function.
        """

        if self._use_bias:
            X = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=-1)

        n_samples, n_features = X.shape
        Y = self.map_y_to_minus_one_plus_one(Y)

        # initialize weights
        self._w = np.zeros(shape=(n_features,), dtype=X.dtype)

        grad = 0
        # loop over epochs
        for e in range(0, self._n_epochs + 1):
            t = 0

            for j in range(n_samples):
                """
                Write HERE the update step.
                """
                t += 1
                if Y[j] * self._w @ X[j] < 1:

                    eta = 1 / (t * self._lambda)
                    if Y[j] * self._w @ X[j] < 1:
                        self._w = (1 - eta * self._lambda) * self._w + eta * Y[j] * X[j]
                    else:
                        self._w = (1 - eta * self._lambda) * self._w

            # grad = self._lambda * self._w - 1 / n_samples * grad  # predict training data
            #
            cur_prediction = np.dot(X, self._w)

            # compute (and print) cost
            cur_loss = self.loss(y_true=Y, y_pred=cur_prediction)

            if verbose:
                print("Epoch {} Loss {}".format(e, cur_loss))


def basic_test():
    x_train, y_train, x_test, y_test = gaussians_dataset(2, [100, 150], [[1, 3], [-4, 8]], [[2, 3], [4, 1]])
    #
    svm_alg = SVM(n_epochs=50, lambDa=0.1, use_bias=True)

    # train
    svm_alg.fit_gd(x_train, y_train, verbose=True)

    # test
    predictions = svm_alg.predict(x_test)

    accuracy = float(np.sum(predictions == y_test)) / y_test.shape[0]
    print('Test accuracy: {}'.format(accuracy))

    plot_pegasos_margin(x_test, y_test, svm_alg)


def people_detection():
    X_img_train, X_feat_train, Y_train, X_img_test, X_feat_test, Y_test = people_dataset(data_path='./data/',
                                                                                         train_split=60)
    model = SVC(C=1.0, kernel='rbf')
    model.fit(X_feat_train, Y_train)
    predictions = model.predict(X_feat_test)
    accuracy = float(np.sum(predictions == Y_test)) / Y_test.shape[0]
    print('Test accuracy: {}'.format(accuracy))

if __name__ == '__main__':
    basic_test()