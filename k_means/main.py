import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons


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


def k_means(points: np.ndarray, n_cluster: int):
    n_points, dim = points.shape
    min_vector = np.array((np.min(points[:, 0]), np.min(points[:, 1])))
    max_vector = np.array((np.max(points[:, 0]), np.max(points[:, 1])))
    cluster_centers = (max_vector - min_vector) * np.random.random_sample((n_cluster, dim)) + min_vector

    plt.scatter(points[:, 0], points[:, 1], c='black')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c=['orange', 'blue'])
    plt.show()
    plt.waitforbuttonpress()
    labels = np.zeros(n_points)

    for k in range(10):
        for i, point in enumerate(points):
            d1 = np.linalg.norm(cluster_centers[0] - point)
            d2 = np.linalg.norm(cluster_centers[1] - point)
            if d1 < d2:
                labels[i] = -1
            else:
                labels[i] = +1

        plt.scatter(points[:, 0], points[:, 1], c=labels)
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c=['green', 'green'])
        plt.show()
        plt.waitforbuttonpress()

        cluster_1 = points[labels == -1]
        cluster_2 = points[labels == 1]

        cluster_1_mean = cluster_1.sum(axis=0) / cluster_1.shape[0]
        cluster_2_mean = cluster_2.sum(axis=0) / cluster_2.shape[0]

        new_centers = np.array([cluster_1_mean, cluster_2_mean])
        if np.isclose(cluster_centers, new_centers).all():
            print("Done! Centers didnt move from previous iteration")
            break
        else:
            cluster_centers[0] = cluster_1_mean
            cluster_centers[1] = cluster_2_mean

    return labels


if __name__ == '__main__':
    # X, y = two_moon_dataset(n_samples=500, noise=0.1)
    #
    # plt.scatter(X[:,0], X[:,1], c=y)
    # plt.show()
    # plt.waitforbuttonpress()
    # plt.close()

    cov_1 = np.diag([2, 1])
    cov_2 = np.diag([1, 2])
    gauss_1 = np.random.multivariate_normal([1, 1], cov_1, size=200)
    gauss_2 = np.random.multivariate_normal([5, 1], cov_2, size=100)

    plt.scatter(gauss_1[:, 0], gauss_1[:, 1], c='orange')
    plt.scatter(gauss_2[:, 0], gauss_2[:, 1], c='blue')
    plt.show()
    plt.waitforbuttonpress()
    # plt.close()

    points = np.vstack((gauss_1, gauss_2))
    np.random.shuffle(points)
    # plt.scatter(points[:, 0], points[:, 1])
    # plt.show()
    # plt.waitforbuttonpress()

    labels = k_means(points, 2)

    fig = plt.figure(layout="constrained")
    ax_array = fig.subplots(1,2, squeeze=False)
    ax_array[0,0].scatter(gauss_1[:, 0], gauss_1[:, 1], c='orange')
    ax_array[0,0].scatter(gauss_2[:, 0], gauss_2[:, 1], c='blue')

    ax_array[0, 1].scatter(points[:, 0], points[:, 1], c=labels)
    fig.show()