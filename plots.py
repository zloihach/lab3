import numpy as np
from matplotlib import pyplot as plt
import dataFunc as df

def plotErrors(errors: np.ndarray):
    plt.plot(errors, 'b-')
    plt.title('Снижение ошибки')
    plt.xlabel('Итерация')
    plt.ylabel('Ошибка')
    plt.grid()
    plt.show()

def plotTask2():
    dataX, datay = df.loadData('ex1data1.txt')
    plt.plot(dataX, datay, 'b.')
    plt.title('Зависимость прибыльности от численности')
    plt.xlabel('Численность')
    plt.ylabel('Прибыльность')
    plt.grid()
    plt.show()


def plotTask6(X: np.matrix,
              y: np.matrix, t: np.matrix):
    x = np.arange(min(X), max(X))
    plt.plot(X, y, 'b.')
    plt.plot(x, t[1, 0] * x + t[0, 0], 'g--')
    plt.title('Зависимость прибыльности от численности')
    plt.xlabel('Численность')
    plt.ylabel('Прибыльность')
    plt.grid()
    plt.show()