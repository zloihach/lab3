# Функция МНК
import numpy as np

def leastSquare(X: np.matrix, y: np.matrix):
    X_ones = np.c_[np.ones((X.shape[0], 1)), X]
    return np.dot(np.linalg.pinv(
        np.dot(X_ones.T, X_ones)),
        np.dot(X_ones.T, y))