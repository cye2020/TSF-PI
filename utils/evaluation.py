import numpy as np
import keras
from datetime import datetime

def MSEs(true, predict):
    return np.average((true- predict)**2)

def MAEs(true, predict):
    return np.average(abs(true - predict))

def Absolute_Error_percentage(true, predict):
    return np.average(abs(true - predict) / true) * 100

def printResult(test_datelist, test_true, test_forecast, target, features):
    date_time = datetime.now()
    date = date_time.strftime("%Y%m%d")
    test_length = test_true[1:, -1].shape[0]
    test_result_list = []
    for i in range(test_length):
        test_result_list.append([test_datelist[- test_length + i], test_true[1:, -1][i], test_forecast[1:][:, -1][i]])
        
    test_yyyymmdd = []
    test_start_y = test_result_list[0][0].year
    test_end_y = test_result_list[-1][0].year
    test_start_m = test_result_list[0][0].month
    test_end_m = test_result_list[-1][0].month
    test_start_m = test_result_list[0][0].day
    test_end_m = test_result_list[-1][0].day

    if (test_start_y == test_end_y):
        for i in range(test_start_m, test_end_m + 1):
            test_yyyymmdd.append(test_start_y * 100 + i)
    else:
        for i in range(test_end_y - test_start_y + 1):
            if (i == 0):
                for j in range(test_start_m, 13):
                    test_yyyymmdd.append((test_start_y + i) * 100 + j)
            elif (i == (test_end_y - test_start_y)):
                for j in range(1, test_end_m + 1):
                    test_yyyymmdd.append((test_start_y + i) * 100 + j)
            else:
                for j in range(1, 13):
                    test_yyyymmdd.append((test_start_y + i) * 100 + j)

    test_month_tf = {}
    for i in range(len(test_yyyymmdd)):
        test_month_tf[test_yyyymmdd[i]] = [0, 0]
    for i in range(len(test_result_list)):
        key = test_result_list[i][0].year * 10000 + test_result_list[i][0].month * 100 + test_result_list[i][0].day
        test_month_tf[key][0] += test_result_list[i][1]
        test_month_tf[key][1] += test_result_list[i][2]
    for i in range(len(test_month_tf)):
        printYear = int(test_yyyymmdd[i] // 10000)
        printMonth = int(test_yyyymmdd[i] % 10000 // 100)
        printDay = int(test_yyyymmdd[i] % 10000 % 100)
        printTrue = int(test_month_tf[test_yyyymmdd[i]][0])
        printForecast = round(test_month_tf[test_yyyymmdd[i]][1])
        printDiff = printTrue - printForecast
        errorRate = round(abs(printTrue - printForecast) / printTrue * 100, 2)
        print('[{0}년 {1}월] {2}일 | 실제 코스피 지수: {3}원    | 예상 코스피 지수: {4}원       | 차이: {5}원       | 절대 오차율: {6}%'.format(printYear, printMonth, printDay, printTrue, printForecast, printDiff, errorRate))