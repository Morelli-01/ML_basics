import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

from LAB2.logreg_stub.my_logreg import LogisticregressionClassiefier
from spectral_clustering.main import spectral_clustering


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

    x_train = np.load('../LAB1/lab01_bayes/mnist/x_train.npy')
    y_train = np.load('../LAB1/lab01_bayes/mnist/y_train.npy')

    x_test = np.load('../LAB1/lab01_bayes/mnist/x_test.npy')
    y_test = np.load('../LAB1/lab01_bayes/mnist/y_test.npy')

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
    mask_train = np.where(y_train == 3, True, False) + np.where(y_train == 4, True, False)
    x_train = x_train[mask_train]
    y_train = y_train[mask_train]
    y_train = np.where(y_train == 3, 0, 1)
    return x_train, y_train, x_test, y_test, label_dict


def two_moon_dataset(n_samples=100, shuffle=True, noise=None, random_state=None):
    """
    Make two interleaving half circles

    A simple toy dataset to visualize clustering and classification
    algorithms.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The total number of points generated.

    shuffle : bool, optional (default=True)
        Whether to shuffle the samples.

    noise : double or None (default=None)
        Standard deviation of Gaussian noise added to the data.

    Read more in the :ref:`User Guide <sample_generators>`.

    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.

    y : array of shape [n_samples]
        The integer labels (0 or 1) for class membership of each sample.
    """
    return make_moons(n_samples=n_samples, shuffle=shuffle, noise=noise, random_state=random_state)


if __name__ == '__main__':
    cov_1 = np.diag([0.5, 1])
    cov_2 = np.diag([1, 0.5])
    gauss_1 = np.random.multivariate_normal([1, 1], cov_1, size=200)
    gauss_2 = np.random.multivariate_normal([5, 1], cov_2, size=250)

    plt.scatter(gauss_1[:, 0], gauss_1[:, 1], c='orange')
    plt.scatter(gauss_2[:, 0], gauss_2[:, 1], c='blue')
    plt.show()
    plt.waitforbuttonpress()
    # plt.close()

    points = np.vstack((gauss_1, gauss_2))
    np.random.shuffle(points)
    x_train_unrolled = points

    # x_train, y_train, x_test, y_test, label_dict = load_mnist()
    #
    # x_train_unrolled = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])

    mean_vector = x_train_unrolled.sum(axis=0) / x_train_unrolled.shape[0]

    x_train_unrolled = x_train_unrolled - mean_vector

    cov = x_train_unrolled.T @ x_train_unrolled

    eigenval, eigenvect = np.linalg.eigh(cov)
    eigenvect = eigenvect.T

    plt.scatter(range(len(eigenval)), eigenval[::-1])
    plt.title('eigenval')
    plt.show()
    plt.waitforbuttonpress()
    plt.close()

    eigenvect_ordered = eigenvect[np.argsort(eigenval)[::-1]]
    eigenval_ordered = np.sort(eigenval)[::-1]

    eigenval_normalized = eigenval_ordered / eigenval_ordered.sum()
    plt.scatter(range(len(eigenval_normalized)), eigenval_normalized)
    plt.title('eigenval normalized')
    plt.show()
    plt.waitforbuttonpress()
    plt.close()

    total_information = 0.0
    k = 1
    for i, val in enumerate(eigenval_normalized):
        total_information += val
        if total_information >= 0.7:
            k = i
            print(f'we got teh 70% of variance taking only {k} eigenvectors')
            break
    if k == 0: k = 1
    projection_matrix = eigenvect_ordered[:, 0:k]

    new_space = x_train_unrolled @ projection_matrix
    labels = np.where(new_space>=0, 1, -1)
    plt.scatter(range(len(new_space)), new_space[:, 0], c=labels)


    # plt.scatter(new_space[:, 0], new_space[:, 1])
    plt.title('new space')
    plt.show()
    plt.waitforbuttonpress()
    plt.close()

    plt.scatter(points[:, 0], points[:, 1], c=labels)
    plt.show()
    plt.waitforbuttonpress()
    plt.close()
