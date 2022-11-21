import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

data = np.matrix(np.loadtxt('ex1data2.txt', delimiter=','))

X = data[:, 0] #площадь
y = data[:, 1] #кол-во комнат
z = data[:, 2] #цена
print(data)


font = {'family': 'Verdana', 'weight': 'normal'}
rc('font', **font)
plt.plot(X, z, 'b.')
plt.title('Зависимость цены от площади')
plt.xlabel('Площадь')
plt.ylabel('Цена')
plt.grid()
plt.show()

plt.plot(y, z, 'b.')
plt.title('Зависимость цены от кол-ва комнат')
plt.xlabel('Кол-во комнат')
plt.ylabel('Цена')
plt.grid()
plt.show()

print('=====================')

#нормализация каждого столбца
def normalize(X):
    X_norm = X
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))
    for i in range(X.shape[1]):
        mu[0, i] = np.mean(X[:, i])
        sigma[0, i] = np.std(X[:, i])
        X_norm[:, i] = (X[:, i] - mu[0, i]) / sigma[0, i]
    return X_norm, mu, sigma


X_norm, mu, sigma = normalize(X)
y_norm, mu, sigma = normalize(y)
z_norm, mu, sigma = normalize(z)


font = {'family': 'Verdana', 'weight': 'normal'}
rc('font', **font)
plt.plot(X_norm, z_norm, 'b.')
plt.title('Зависимость цены от площади')
plt.xlabel('Площадь')
plt.ylabel('Цена')
plt.grid()
plt.show()

font = {'family': 'Verdana', 'weight': 'normal'}
rc('font', **font)
plt.plot(y_norm, z_norm, 'b.')
plt.title('Зависимость цены от кол-ва комнат')
plt.xlabel('Кол-во комнат')
plt.ylabel('Цена')
plt.grid()
plt.show()


#градиентный спуск
def gradient_descent(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        theta = theta - (alpha / m) * (X.T @ (X @ theta - y))
        J_history[i] = compute_cost(X, y, theta)
    return theta, J_history

def compute_cost(X, y, theta):
    m = y.size
    J = 0
    J = (1 / (2 * m)) * np.sum(np.square(X @ theta - y))
    return J

#добавляем столбец из единиц
X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
y = np.concatenate([np.ones((y.shape[0], 1)), y], axis=1)
z = np.concatenate([np.ones((z.shape[0], 1)), z], axis=1)

#параметры
alpha = 0.01
num_iters = 400

#запускаем градиентный спуск
theta, J_history = gradient_descent(X, z, np.zeros((2, 1)), alpha, num_iters)
theta1, J_history1 = gradient_descent(y, z, np.zeros((2, 1)), alpha, num_iters)

#выводим график
plt.plot(np.arange(J_history.size), J_history, 'b.')
plt.title('График зависимости функции от итераций')
plt.xlabel('Итерации')
plt.ylabel('Функция')
plt.grid()
plt.show()


#метод наименьших квадратов

def normal_equation(X, y):
    theta = np.zeros((X.shape[1], 1))
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y
    return theta

theta2 = normal_equation(X, z)
theta3 = normal_equation(y, z)

print('theta = ', theta)
print('theta1 = ', theta1)
print('theta2 = ', theta2)
print('theta3 = ', theta3)

# #предсказание
# def predict(X, theta):
#     return X @ theta
#
# #площадь
# area = 100
# #кол-во комнат
# rooms = 3
#
# #нормализуем
# area_norm = (area - mu[0, 0]) / sigma[0, 0]
# rooms_norm = (rooms - mu[0, 1]) / sigma[0, 1]
#
#
# #добавляем столбец из единиц
# area_norm = np.concatenate([np.ones((area_norm.shape[0], 1)), area_norm], axis=1)
# rooms_norm = np.concatenate([np.ones((rooms_norm.shape[0], 1)), rooms_norm], axis=1)
#
# #предсказываем
# price = predict(area_norm, theta)
# price1 = predict(rooms_norm, theta1)
# price2 = predict(area_norm, theta2)
# price3 = predict(rooms_norm, theta3)
#
# print('price = ', price)
# print('price1 = ', price1)
# print('price2 = ', price2)
# print('price3 = ', price3)
#
# #выводим график
# plt.plot(X_norm, z_norm, 'b.')
# plt.plot(X_norm, predict(X, theta), 'r-')
# plt.title('Зависимость цены от площади')
# plt.xlabel('Площадь')
# plt.ylabel('Цена')
# plt.grid()
#plt.show()