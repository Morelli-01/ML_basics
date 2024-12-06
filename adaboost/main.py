import numpy as np

from weakClassifier import load_mnist, WeakClassifier


def create_n_weakClassifier(n=100):
    x_train, y_train, x_test, y_test, label_dict = load_mnist()
    mask_train = np.where(y_train == 8, True, False) + np.where(y_train == 9, True, False)
    x_train = x_train[mask_train]
    y_train = y_train[mask_train]
    y_train = np.where(y_train == 9, 1, -1)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
    x_train = x_train[:1000, :]
    y_train = y_train[:1000]

    z = x_train.shape[0]
    weights = np.ones(x_train.shape[0])
    weights = weights * 1 / z

    classifiers = []
    for i in range(n):
        mask = np.random.choice(a=range(0, z), size=z, replace=True, p=weights)
        mask = np.unique(mask)
        x_sampled = x_train[mask]
        y_sampled = y_train[mask]
        sampled_weights = weights[mask]
        accuracy = 0.0

        while accuracy < 0.5:
            wc = WeakClassifier()
            wc.fit(x_sampled, y_sampled)

            y_predicted = wc.predict(x_sampled)
            accuracy = np.sum(np.where(y_predicted == y_sampled, 1, 0)) / z
            print(f'accuracy is {accuracy}')

            if accuracy > 0.5:
                mask = np.unique(mask)
                x_sampled = np.unique(x_sampled)


                predicted_mask = np.where(y_predicted != y_sampled, True, False)
                error = np.sum(sampled_weights[predicted_mask])
                print(f'error is {error}')
                alpha = np.log((1 - error) / error) / 2
                for j in range(z):
                    if y_predicted[j] == y_sampled[j]:
                        weights[mask[j]] = np.exp(-alpha)
                    else:
                        weights[mask[j]] = np.exp(alpha)

                weights = weights / np.sum(weights)
                print(f'weights sum {np.sum(weights)}')

        classifiers.append(wc)

    return classifiers


if __name__ == '__main__':
    classifiers = create_n_weakClassifier(n=100)
