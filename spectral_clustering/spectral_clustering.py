import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
import networkx as nx


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


def spectral_clustering(points: np.array, n_cluster: int, threeshold):
    n_points, dim = points.shape
    gamma = 2.5
    adjacency_matrix = np.zeros((n_points, n_points))
    for i, point_i in enumerate(points):
        for j, point_j in enumerate(points):
            if i == j:
                continue
            if np.linalg.norm(point_j - point_i) < threeshold:
                # adjacency_matrix[i, j] = 1
                adjacency_matrix [i, j] = np.exp(-gamma* np.linalg.norm(point_j - point_i))
    degree_matrix = np.zeros((n_points, n_points))
    for i in range(n_points):
        degree_matrix[i, i] = adjacency_matrix.sum(axis=1)[i]

    laplacian_matrix = degree_matrix - adjacency_matrix
    G = nx.from_numpy_array(adjacency_matrix)

    # Disegna il grafo
    pos = nx.spring_layout(G)  # Puoi utilizzare altri layout a seconda delle tue preferenze
    nx.draw(G, pos, with_labels=False, node_size=20, node_color="skyblue", font_size=8, font_color="black",
            font_weight="bold", edge_color="black", linewidths=1, font_family="sans-serif")

    plt.show()
    plt.waitforbuttonpress()
    plt.close()

    eigenval, eigenvect = np.linalg.eig(laplacian_matrix)
    sorted_indices = np.argsort(eigenval)
    eigenval = eigenval[sorted_indices]
    eigenvect = eigenvect[:, sorted_indices]
    eigenvect = eigenvect.T


    new_feature = eigenvect[1]
    labels = np.zeros(500)
    labels = np.where(new_feature > 0, 1, 0)
    plt.scatter(points[:, 0], points[:, 1], c=labels)
    plt.show()
    plt.waitforbuttonpress()
    plt.close()

    new_feature2 = eigenvect[2]
    plt.scatter(new_feature, new_feature2, c=labels)
    plt.show()
    plt.waitforbuttonpress()
    plt.close()

    eigengap, k = (0, 0)
    eigenval_sorted = np.sort(eigenval)
    plt.scatter(range(len(eigenval_sorted)), eigenval_sorted)
    plt.title('EIgengap Heuristic')
    plt.show()
    plt.waitforbuttonpress()
    plt.close()
    for i, val in enumerate(eigenval_sorted):
        if i == 0:
            continue
        if np.abs(eigenval_sorted[i]-eigenval_sorted[i-1]) > eigengap:
            eigengap = np.abs(eigenval_sorted[i]-eigenval_sorted[i-1])
            k = i
    print(f'optiman number of cluster based on heuristic eigengap is {k}')
    return np.array((new_feature, new_feature2))


if __name__ == '__main__':
    # for twomoon
    X, y = two_moon_dataset(n_samples=500, noise=0.1)

    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()
    plt.waitforbuttonpress()
    plt.close()
    points = X
    # for gaussian
    # cov_1 = np.diag([0.5, 1])
    # cov_2 = np.diag([1, 0.5])
    # gauss_1 = np.random.multivariate_normal([1, 1], cov_1, size=200)
    # gauss_2 = np.random.multivariate_normal([5, 1], cov_2, size=100)
    #
    # plt.scatter(gauss_1[:, 0], gauss_1[:, 1], c='orange')
    # plt.scatter(gauss_2[:, 0], gauss_2[:, 1], c='blue')
    # plt.show()
    # plt.waitforbuttonpress()
    # plt.close()
    #
    # points = np.vstack((gauss_1, gauss_2))
    # X = points

    threeshold = 0.1
    for i in range(20):
        threeshold += 0.1
        print(f'threeshold = {threeshold}')
        new_space = spectral_clustering(points, 2, threeshold)
        new_space = new_space.T
        # k_means(new_space, 2)
