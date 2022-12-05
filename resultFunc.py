# Функция предсказания
import numpy as np

def findResult(X_res: np.matrix,
               theta_res, data_norm=np.matrix([0])):
    if X_res.shape[1] >= theta_res.shape[0]:
        return None
    y_res = theta_res[0, 0]
    if data_norm.shape == (1, 1):
        for i in range(X_res.shape[1]):
            y_res += theta_res[i + 1, 0] * X_res[0, i]
        return y_res
    x_res_tmp = X_res.copy()
    for i in range(x_res_tmp.shape[1]):
        x_res_tmp[0, i] -= data_norm[i, 0]
        x_res_tmp[0, i] /= data_norm[i, 1]
    for i in range(X_res.shape[1]):
        y_res += theta_res[i + 1, 0] * x_res_tmp[0, i]
    y_res *= data_norm[-1, 1]
    y_res += data_norm[-1, 0]
    return y_res

# Функция вычисления стоимости
def computeCost(X_cost: np.matrix, y_cost: np.matrix,
                theta_cost: np.matrix):
    m = X_cost.shape[0]
    h_x = X_cost * theta_cost
    cost = (1 / (2 * m) * np.power(h_x - y_cost, 2)).sum()
    return cost

# Функция градиентного спуска
def gradient_descent(X_grad: np.matrix, y_grad: np.matrix,
                     alpha: float, iterations: int):
    m, n = X_grad.shape
    theta_grad = np.ones((n, 1))
    theta_grad[0, 0] = 0
    j_theta = np.zeros((iterations, 1))
    temp_theta = theta_grad
    for i in range(iterations):
        theta_grad = temp_theta.copy()
        j_theta[i] = computeCost(X_grad, y_grad, theta_grad)
        temp_theta = theta_grad - alpha * (1 / m) *\
                     np.dot(X_grad.T, X_grad * theta_grad - y_grad)
    return j_theta, theta_grad
