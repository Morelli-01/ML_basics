import numpy as np

total_example = 100

if __name__ == '__main__':
    distribution = np.zeros(total_example)
    for i in range(total_example):
        distribution[i] = np.random.randint(1, 2)

    class_1 = (np.where(distribution == 1, 1, 0).sum() + np.finfo(np.float32).eps)/100
    class_2 = (np.where(distribution == 2, 1, 0).sum() + np.finfo(np.float32).eps)/100
    class_3 = (np.where(distribution == 3, 1, 0).sum() + np.finfo(np.float32).eps)/100

    entropy = -(class_1 * np.log(class_1) + class_2 * np.log(class_2) + class_3 * np.log(class_3))
