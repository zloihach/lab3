import numpy as np

import resultFunc as rf
import dataFunc as df
import plots as myplots
import leastSquare as ls

def oneRegress():
    X, y = df.loadData('ex1data1.txt')
    myplots.plotTask2()
    X_ones = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.matrix('[1; 2]')
    # print("Стоимость при 'theta' = [1,2]: {}".format(str(computeCost(X_ones, y, theta))))
    r, t = rf.gradient_descent(X_ones, y, 0.02, 500)
    print("'theta' найденная градиентным спуском: " + str(t).replace("\n", ""))
    myplots.plotErrors(r)
    # print(
    #     "Приблизительная прибыль при численности населения 8,5781: {}".format(str(findResult(np.matrix((8.5781)), t))))
    # print(
    #     "Приблизительная прибыль при численности населения 3.4567: {}".format(str(findResult(np.matrix((3.4567)), t))))
    myplots.plotTask6(X, y, t)

def twoRegress():
    data2 = df.loadData('ex1data2.txt')
    norm_data, norm = df.normalize(np.c_[data2[0], data2[1]])
    X2, y2 = norm_data[:, 0:-1], norm_data[:, -1]
    X_ones2 = np.c_[np.ones((X2.shape[0], 1)), X2]
    r2, t2 = rf.gradient_descent(X_ones2, y2, 0.02, 500)
    last = ls.leastSquare(data2[0], data2[1])

    quartile_area1 = input("Введите площадь квартиры для первого рассчета: ")
    number_of_rooms1 = input("Введите количество комнат для первого рассчета: ")

    quartile_area2 = input("Введите площадь квартиры для второго рассчета: ")
    number_of_rooms2 = input("Введите количество комнат для второго рассчета: ")

    print(
        "Приблезительная стоимость квартиры с площадью {} и {} комнатоай(ами): {} МНК и {} градиентный спуск".format(
            str(quartile_area1),
            str(number_of_rooms1),
            str(rf.findResult(np.matrix([quartile_area1, number_of_rooms1], dtype=np.float64), last)),
            str(rf.findResult(np.matrix([quartile_area1, number_of_rooms1], dtype=np.float64), t2, norm))))
    print(
        "Приблезительная стоимость квартиры с площадью {} и {} комнатой(ами): {} МНК и {} градиентный спуск".format(
            str(quartile_area2),
            str(number_of_rooms2),
            str(rf.findResult(np.matrix([quartile_area2, number_of_rooms2], dtype=np.float64), last)),
            str(rf.findResult(np.matrix([quartile_area2, number_of_rooms2], dtype=np.float64), t2, norm))))