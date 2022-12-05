# Загрузка данных из файла
import numpy as np

def loadData(file: str):
    data = np.matrix(np.
                     loadtxt(file, delimiter=','))
    return data[:, 0:-1], data[:, -1]


def normalize(norm_data: np.matrix):
    tmp = np.zeros((norm_data.shape[1], 2))
    tmp[:, 0] = np.mean(norm_data, axis=0)
    tmp[:, 1] = np.std(norm_data, axis=0)
    norm_data -= tmp[:, 0]
    norm_data /= tmp[:, 1]
    return norm_data, tmp