import numpy as np


class NaiveBayesClassifier:
    def __init__(self):
        self._classes = None
        self._n_classes = 0

        self._eps = np.finfo(np.float32).eps

        # array of classes prior probabilities
        self._class_priors = []

        # array of probabilities of a pixel being active (for each class)
        self._pixel_probs_given_class = []

    def fit(self, x, y):
        self._classes = np.unique(y)
        self._n_classes = len(self._classes)

        for i in range(self._n_classes):
            tmp = 0
            matrix_tmp = np.zeros((x.shape[1], x.shape[2]))
            for index, j in enumerate(y):
                if j == i:
                    matrix_tmp += x[index]
                    tmp += 1

            self._pixel_probs_given_class.append(matrix_tmp / tmp)
            self._class_priors.append(tmp / y.shape[0])

    def predict(self, X):
        y_predicted = np.zeros((X.shape[0]))
        for i, x in enumerate(X):
            posterior_buffer = np.zeros((self._n_classes))
            for j in range(self._n_classes):
                prior = self._class_priors[j]
                pixel_likelyhood = self._pixel_probs_given_class[j]

                log_likelyhood = x * pixel_likelyhood + (1.0 - pixel_likelyhood) * (1 - x)
                log_likelyhood = np.sum(np.log(log_likelyhood + self._eps)) + np.log(prior)
                posterior_buffer[j] = log_likelyhood

            sum = np.sum(np.exp(posterior_buffer))
            for j in range(self._n_classes):
                posterior_buffer[j] = np.exp(posterior_buffer[j]) / sum

            y_predicted[i] = np.argmax(posterior_buffer)

        return y_predicted


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

    x_train = np.load('mnist/x_train.npy')
    y_train = np.load('mnist/y_train.npy')

    x_test = np.load('mnist/x_test.npy')
    y_test = np.load('mnist/y_test.npy')

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


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, label_dict = load_mnist()
    nbc = NaiveBayesClassifier()
    nbc.fit(x_train, y_train)
    y_predicted = nbc.predict(x_test)
    accuracy = np.sum(np.uint8(y_predicted == y_test)) / len(y_test)
    print(f"Bayes Accuracy: {accuracy}")
