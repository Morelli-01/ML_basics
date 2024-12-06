from my_logreg import load_mnist, LogisticregressionClassiefier
import numpy as np

if __name__ == '__main__':
    x_train, y_train, x_test, y_test, label_dict = load_mnist()

    classifier = []
    for i in label_dict:
        classifier_i = []
        for j in range(i + 1, len(label_dict)):
            x_train, y_train, x_test, y_test, label_dict = load_mnist()

            mask_train = np.where(y_train == i, True, False) + np.where(y_train == j, True, False)
            x_train = x_train[mask_train]
            y_train = y_train[mask_train]
            y_train = np.where(y_train == i, 0, 1)
            # mask_test = np.where(y_test == i, True, False) + np.where(y_test == j, True, False)
            # x_test = x_test[mask_test]
            # y_test = y_test[mask_test]
            # y_test = np.where(y_test == i, 0, 1)

            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
            lgc = LogisticregressionClassiefier()
            lgc.fit(x_train, y_train, n_epoch=6)
            # y_predicted = lgc.predict(x_test)
            # accuracy = np.sum(y_predicted == y_test) / y_test.shape[0]
            # print(f'Accuracy: {accuracy}')
            classifier_i.append(lgc)

        classifier.append(classifier_i)

    x_train, y_train, x_test, y_test, label_dict = load_mnist()

    # mask_train = np.where(y_train == i, True, False) + np.where(y_train == j, True, False)
    # x_train = x_train[mask_train]
    # y_train = y_train[mask_train]
    # y_train = np.where(y_train == i, 0, 1)
    mask_test = np.where(y_test == i, True, False) + np.where(y_test == j, True, False)
    # x_test = x_test[mask_test]
    # y_test = y_test[mask_test]
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

    # y_test = np.where(y_test == i, 0, 1)
    voting_buffer = np.zeros((x_test.shape[0], len(label_dict)))
    for i in label_dict:
        for j in range(len(label_dict) - (i+1)):
            predictions = classifier[i][j].predict(x_test)
            for index, predict in enumerate(predictions):
                if predict == 1.0:
                    voting_buffer[index][i + j + 1] += 1
                else:
                    voting_buffer[index][i] += 1

    predictions = np.zeros(x_test.shape[0])
    for i, predict in enumerate(voting_buffer):
        predictions[i] = np.argmax(predict)

    accuracy = np.sum(np.uint8(predictions == y_test)) / len(y_test)
    print('Accuracy: {}'.format(accuracy))