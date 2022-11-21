import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

data = np.matrix(np.loadtxt('ex1data1.txt', delimiter=','))

X = data[:, 0]
y = data[:, 1]
print(data)

font = {'family': 'Verdana', 'weight': 'normal'}
rc('font', **font)
plt.plot(X, y, 'b.')
plt.title('Зависимость прибыльности от численности')
plt.xlabel('Численность')
plt.ylabel('Прибыльность')
plt.grid()
plt.show()



print('=====================')
m = X.shape[0] # количество элементов в X (количество городов)
X_ones = np.c_[np.ones((m, 1)), X] # добавляем единичный столбец к X
theta = np.matrix('[1; 2]') # коэффициенты theta представляют собой вектор-столбец из 2 элементов
# альтернативные способы создания вектора-столбца theta:
# theta = np.matrix([1, 2]).reshape(2, 1)
# theta = np.matrix([[1], [2]])
# theta = np.matrix([1, 2]).transpose()
h_x = X_ones * theta # так можно вычислить значение гипотезы для всех городов сразу (подумайте
#$почему и поэкспериментируйте с этим кодом в консоли Python).
print(h_x)


def compute_cost(X, y, theta):
    m = len(y)
    h_x = X * theta
    cost = (1 / (2 * m)) * np.sum(np.square(h_x - y))
    return cost

cost = compute_cost(X_ones, y, theta)
print(cost)


# def gradient_descent(X, y, theta, alpha, num_iters):
#     m = len(y)
#     J_history = np.zeros((num_iters, 1))
#     for i in range(num_iters):
#         h_x = X * theta
#         theta = theta - (alpha / m) * (X.transpose() * (h_x - y))
#         J_history[i] = compute_cost(X, y, theta)
#     return theta, J_history

#две реализации с одинаковыми результатами, не знаю в чем смысл добавлять temp_theta :(
def gradient_descent(X, y, theta, alpha, num_iters):
    #temp_theta
    temp_theta = np.matrix(np.zeros(theta.shape))
    m = len(y)
    J_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        h_x = X * theta
        for j in range(len(theta)):
            temp_theta[j] = theta[j] - (alpha / m) * (np.sum(np.multiply((h_x - y), X[:, j])))
        theta = temp_theta
        J_history[i] = compute_cost(X, y, theta)
    return theta, J_history


theta, J_history = gradient_descent(X_ones, y, theta, 0.02, 500)
print(theta)


plt.plot(J_history[10:], 'b.')
plt.title('Cost function')
plt.grid()
plt.show()


print('=====================MIN/MAX=====================')

minx = min(X)
maxx = max(X)
print(minx)
print(maxx)

np.arange(minx, maxx, 0.1)
plt.plot(X, y, 'b.')
plt.title('Зависимость прибыльности от численности')
plt.xlabel('Численность')
plt.ylabel('Прибыльность')
plt.grid()
plt.plot(X, X_ones * theta, 'g--')
plt.plot(minx, theta[0] + theta[1] * minx, 'g--')
plt.show()

