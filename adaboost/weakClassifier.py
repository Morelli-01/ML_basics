import numpy as np


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

    x_train = np.load('../bayesian_cls/mnist/x_train.npy')
    y_train = np.load('../bayesian_cls/mnist/y_train.npy')

    x_test = np.load('../bayesian_cls/mnist/x_test.npy')
    y_test = np.load('../bayesian_cls/mnist/y_test.npy')

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

    # x_train = np.float32(x_train) / 255.
    # x_train[x_train >= threshold] = 1
    # x_train[x_train < threshold] = 0
    #
    # x_test = np.float32(x_test) / 255.
    # x_test[x_test >= threshold] = 1
    # x_test[x_test < threshold] = 0

    return x_train, y_train, x_test, y_test, label_dict


class WeakClassifier:
    def __init__(self, ):
        self.dimensionality = 0
        self.threshold = 0

    def fit(self, X: np.array, Y: np.array):
        n, d = X.shape
        labels = np.unique(Y)
        while np.min(X[:, self.dimensionality]) == np.max(X[:, self.dimensionality]):
            self.dimensionality = np.random.choice(a=range(0, d))

        self.threshold = np.random.choice(a=range(np.min(X[:, self.dimensionality]), np.max(X[:, self.dimensionality])))

        self.above_label = np.random.choice(labels)
        self.under_label = -self.above_label

        # self.threshold = np.random.randn()

    def predict(self, X: np.array):
        y_predict = np.where(X[:, self.dimensionality] > self.threshold, self.above_label, self.under_label)
        return y_predict


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, label_dict = load_mnist()
    mask_train = np.where(y_train == 8, True, False) + np.where(y_train == 9, True, False)
    x_train = x_train[mask_train]
    y_train = y_train[mask_train]
    y_train = np.where(y_train == 9, 1, -1)
    mask_test = np.where(y_test == 8, True, False) + np.where(y_test == 9, True, False)
    x_test = x_test[mask_test]
    y_test = y_test[mask_test]
    y_test = np.where(y_test == 9, 1, -1)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

    accuracy = 0.0
    while accuracy < 0.5:
        wc = WeakClassifier()
        wc.fit(x_train, y_train)

        y_predicted = wc.predict(x_test)
        accuracy = np.sum(np.where(y_predicted == y_test, 1, 0)) / y_test.shape[0]
        print(f'accuracy is {accuracy}')