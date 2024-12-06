import numpy as np

from data_io import load_got_dataset


def load_mnist(threshold=0.5):
    """
    Loads mnist (original, with digits).

    Returns
    -------
    tuple:
        x_train with shape(n_train_samples, h, w)
        y_train with shape(n_train_samples,)
        x_test with shape(n_test_samples, h, w)
        y_test with shape(n_test_samples,)
    """

    x_train = np.load('../../LAB1/lab01_bayes/mnist/x_train.npy')
    y_train = np.load('../../LAB1/lab01_bayes/mnist/y_train.npy')

    x_test = np.load('../../LAB1/lab01_bayes/mnist/x_test.npy')
    y_test = np.load('../../LAB1/lab01_bayes/mnist/y_test.npy')

    label_dict = {i: str(i) for i in range(0, 10)}

    """
    Loads MNIST data (either digits or fashion) and returns it binarized.

    Parameters
    ----------
    threshold: float
        a threshold in [0, 1] to binarize w.r.t.

    Returns
    -------
    tuple:
        x_train with shape(n_train_samples, h, w)
        y_train with shape(n_train_samples,)
        x_test with shape(n_test_samples, h, w)
        y_test with shape(n_test_samples,)
    """

    x_train = np.float32(x_train) / 255.
    x_train[x_train >= threshold] = 1
    x_train[x_train < threshold] = 0

    x_test = np.float32(x_test) / 255.
    x_test[x_test >= threshold] = 1
    x_test[x_test < threshold] = 0

    return x_train, y_train, x_test, y_test, label_dict


class LogisticregressionClassiefier:
    def __init__(self):
        """ Constructor method """
        # weights placeholder
        self._w = None

        self.alpha_start = 10
        self.beta = 0.5
        self._classes = [0, 1]

    def fit(self, X, Y, n_epoch=10000, threeshold=0.001):
        dim = X.shape[1]
        count = X.shape[0]
        self._w = np.random.rand(dim)
        d = np.zeros(dim)
        loss = self.loss_func(self._w, X, Y)
        for i in range(n_epoch):
            for x, y in zip(X, Y):
                d += (y - (np.exp(self._w.T @ x) / (1 + np.exp(self._w.T @ x)))) * x

            d /= -count
            alpha = self.alpha_start

            while self.loss_func(self._w - alpha * d, X, Y) > loss:
                alpha = alpha * self.beta
            self._w = self._w - alpha * d
            loss = self.loss_func(self._w, X, Y, verbose=False)
            if np.linalg.norm(d) < threeshold:
                print(f'we stopped at epoch: {i}')
                break

    def predict(self, X):
        y_predict = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            tmp = self._w @ x
            f = np.exp(tmp) / (1 + np.exp(tmp))
            if f > 0.5:
                y_predict[i] = self._classes[1]
            else:
                y_predict[i] = self._classes[0]

        return y_predict

    def loss_func(self, w, X, Y, verbose=False):
        count = Y.shape[0]
        loss = 0
        for x, y in zip(X, Y):
            loss += y * (w.T @ x) - np.log(1 + np.exp(w.T @ x))
        loss /= -count
        if verbose:
            print(f'the loss is {loss}')
        return loss


if __name__ == '__main__':
    # x_train, y_train, train_names, x_test, y_test, test_names, feature_names = load_got_dataset(path='data/got.csv',
    #                                                                                             train_split=0.8)
    x_train, y_train, x_test, y_test, label_dict = load_mnist()

    mask_train = np.where(y_train == 3, True, False) + np.where(y_train == 4, True, False)
    x_train = x_train[mask_train]
    y_train = y_train[mask_train]
    y_train = np.where(y_train == 3, 0, 1)
    mask_test = np.where(y_test == 3, True, False) + np.where(y_test == 4, True, False)
    x_test = x_test[mask_test]
    y_test = y_test[mask_test]
    y_test = np.where(y_test == 3, 0, 1)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
    lgc = LogisticregressionClassiefier()
    # lgc._classes = [3,4]
    lgc.fit(x_train, y_train, n_epoch=100)
    y_predicted = lgc.predict(x_test)
    accuracy = np.sum(y_predicted == y_test) / y_test.shape[0]
    print(f'Accuracy: {accuracy}')
