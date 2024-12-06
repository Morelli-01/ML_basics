from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


class Kernel:
    def __init__(self, stride, padding, kernel):
        self.size = kernel.shape[0]
        self.stride = stride
        self.padding = padding
        self.kernel = kernel

    def convolve(self, img_array: np.array):
        if len(img_array.shape) < 3:
            c = 0
        else:
            c = img_array.shape[2]

        result = np.zeros((int((img_array.shape[0] - self.size + 2 * self.padding) / self.stride + 1),
                           int((img_array.shape[1] - self.size + 2 * self.padding) / self.stride + 1)))
        for i1, i in enumerate(range(0, img_array.shape[0], self.stride)):
            for j1, j in enumerate(range(0, img_array.shape[1], self.stride)):
                if c != 0:
                    sub_matrix = img_array[i:self.size + i, j:self.size + j, :]
                else:
                    sub_matrix = img_array[i:self.size + i, j:self.size + j]
                if sub_matrix.shape[0] < self.size or sub_matrix.shape[1] < self.size:
                    break
                try:
                    tmp = np.zeros((self.size, self.size))
                    if c != 0:
                        for k in range(c):
                            tmp += sub_matrix[:, :, k] * self.kernel
                    else:
                        tmp = sub_matrix[:, :] * self.kernel

                    result[i1, j1] = tmp.sum()
                except IndexError as e:
                    print(e)
                    return result
        return result


class Pooling:
    def __init__(self, size, stride):
        self.size = size
        self.stride = stride

    def max_pooling(self, img_array: np.array):
        result = np.zeros((int((img_array.shape[0] - self.size) / self.stride + 1),
                           int((img_array.shape[1] - self.size) / self.stride + 1)))
        for i1, i in enumerate(range(0, img_array.shape[0], self.stride)):
            for j1, j in enumerate(range(0, img_array.shape[1], self.stride)):
                sub_matrix = img_array[i:self.size + i, j:self.size + j]
                if sub_matrix.shape[0] < self.size or sub_matrix.shape[1] < self.size:
                    break
                try:
                    result[i1, j1] = np.max(sub_matrix[:, :])
                except IndexError as e:
                    print(e)
                    return result
        return result


def gaussian_kernel(size, center, sigma=1.0):
    x, y = np.mgrid[0:size, 0:size]
    pos = np.dstack((x, y))
    rv = multivariate_normal(center, cov=sigma)
    return rv.pdf(pos)


if __name__ == '__main__':
    immagine = Image.open('../data/kirby.jpg')
    img_array = np.array(immagine)
    # img_array = img_array[400:1350, 550:1450, :]
    plt.imshow(img_array, cmap='Greys')
    plt.show()
    plt.waitforbuttonpress()
    plt.close()

    kernel_mat_X = np.array(((-1, 0, 1),
                           (-1, 0, 1),
                           (-1, 0, 1)))*1/-6
    kernel_mat_Y = np.array(((-1, -1, -1),
                           (0, 0, 0),
                           (1, 1, 1))) * 1 / -6
    # kernel_mat = gaussian_kernel(50, (25, 25), sigma=100000.0)
    kernelX = Kernel(stride=1, padding=0, kernel=kernel_mat_X)
    # pooling1 = Pooling(2, 8)

    result1 = kernelX.convolve(img_array)
    plt.imshow(result1, cmap='Greys')
    plt.show()
    plt.waitforbuttonpress()
    plt.close()

    kernelY = Kernel(stride=1, padding=0, kernel=kernel_mat_Y)
    # pooling1 = Pooling(2, 8)

    result2 = kernelY.convolve(img_array)

    plt.imshow(result2, cmap='Greys')
    plt.show()
    plt.waitforbuttonpress()
    plt.close()

    threshold = 100
    plt.imshow(np.where(result1+result2 > threshold, 255, 0 ), cmap='binary')
    plt.show()
    plt.waitforbuttonpress()
    plt.close()
    #
    # result = kernel.convolve(result)
    # plt.imshow(result)
    # plt.show()
    # plt.waitforbuttonpress()
    # plt.close()
    #
    # result = pooling1.max_pooling(result)
    # plt.imshow(result)
    # plt.show()
    # plt.waitforbuttonpress()
    # plt.close()

    # kernel = Kernel(stride=10, padding=0, kernel=kernel_mat.T)
    # result = kernel.convolve(img_array)
    # plt.imshow(result)
    # plt.show()
    # plt.waitforbuttonpress()
    # plt.close()
