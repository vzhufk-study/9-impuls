import copy
from math import sin, cos
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def function(equation, t, results, variables):
    equationLocal = equation.replace(variables[0], 't');
    i = 0
    for variable in variables[1]:
        equationLocal = equationLocal.replace(variable, 'results[' + str(i) + ']')
        i += 1
    return eval(equationLocal)


def solveSystem(equations=None, variables=None, values=None, interval=None, e=0.01, N=2, maxN=9999999999,
                      equationsExact=None, c=None):
    if equations is None:
        equations = ["x", "x + y", "x + y + z"]
    if variables is None:
        variables = ["t", ["x", "y", "z"]]
    if values is None:
        values = [[3, None], [2, None], [1, None]]
    if interval is None:
        interval = [0, 10]
    if c is None:
        c = [1, 1, 1]

    n = N
    h = (interval[1] - interval[0]) / n
    t_new = [interval[0]]
    result_start = []
    results_start = []
    flag = 1
    size = len(variables[1])

    for value in values:
        result_start.append([value[0]])
        results_start.append(value[0])

    result_new = copy.deepcopy(result_start)
    results_new = copy.deepcopy(results_start)

    for i in range(0, n):
        results = results_new
        results_new = []



        t_new.append(t_new[i] + flag * h)
        for j in range(0, size):
            
            a = t_new[i] + h/2
            b = result_new[j][i] + (h/2)*t_new[i]
            results[1] = b;
            temp_result = result_new[j][i] + flag * h * function(equations[j], a, results, variables)
            result_new[j].append(temp_result)
            results_new.append(temp_result)
    t_list = t_new
    result = result_new

    index = 0
    while index < n / 2 + 1 and n < maxN:
        t_list = t_new
        result = result_new
        t_new = [t_new[0]]
        result_new = copy.deepcopy(result_start)
        results_new = copy.deepcopy(results_start)

        n *= 2
        h = (interval[1] - interval[0]) / n

        for i in range(0, n):
            results = results_new
            results_new = []

            t_new.append(t_new[i] + flag * h)
            for j in range(0, size):
                temp_result = result_new[j][i] + flag * h * function(equations[j], t_new[i], results, variables)
                result_new[j].append(temp_result)
                results_new.append(temp_result)

        index = 0
        good = True
        while index < len(t_list) and good:
            j = 0
            while j < size and abs(result[j][index] - result_new[j][index * 2]) < e:
                j += 1
            if j == size:
                index += 1
            else:
                good = False


    result_exact = []
    if equationsExact is not None:
        equationsLocal = copy.deepcopy(equationsExact)
        for j in range(0, size):
            equationsLocal[j] = equationsLocal[j].replace(variables[0], 't')
            equationsLocal[j] = equationsLocal[j].replace('exp(', 'np.exp(')
            result_exact.append([])

            for t in t_list:
                result_exact[j].append(eval(equationsLocal[j]))

    return n / 2, h * 2, t_list, result, result_exact


def solveImpulsesCauchySystem(equations=None, variables=None, values=None, interval=None, e=0.01, N=2, maxN=9999999999,
                              equationsExact=None, equationsFindC=None, impulses=3, delta=None):
    if equations is None:
        equations = ["x", "x + y", "x + y + z"]
    if variables is None:
        variables = ["t", ["x", "y", "z"]]
    if values is None:
        values = [[3, None], [2, None], [1, None]]
    if interval is None:
        interval = [0, 10]
    if delta is None:
        delta = ["0.3 * x", "2 / y", "j / 4 + z"]

    intervalLocal = [0, interval[0]]
    valuesLocal = copy.deepcopy(values)
    equationsFindCLocal = []
    deltaLocal = []
    result = [[], [], [], [], []]

    for j in range(0, len(equationsFindC)):
        temp = equationsFindC[j].replace(variables[0], 'intervalLocal[0]')
        temp = temp.replace('exp', 'np.exp')
        tempDelta = delta[j].replace(variables[0], 'intervalLocal[0]')

        i = 0
        for variable in variables[1]:
            temp = temp.replace(variable + ' ', 'valuesLocal[' + str(i) + '][0]')
            tempDelta = tempDelta.replace(variable + ' ', 'valuesLocal[' + str(i) + '][0]')
            tempDelta = tempDelta.replace(' ' + variable, 'valuesLocal[' + str(i) + '][0]')
            i += 1

        equationsFindCLocal.append(temp)
        deltaLocal.append(tempDelta)

    for j in range(0, impulses + 1):
        intervalLocal[0] = intervalLocal[1]
        intervalLocal[1] = intervalStart[0] + (intervalStart[1] - intervalStart[0]) / (impulses + 1) * (j + 1)
        c = []
        for i in range(0, len(equationsFindCLocal)):
            c.append(eval(equationsFindCLocal[i]))

        temp = solveSystem(equations, variables, valuesLocal, intervalLocal, e, N, maxN,
                                 equationsExact, c)

        for i in range(0, 2):
            result[i].append(temp[i])
        result[2].extend(temp[2])
        for i in range(3, 5):
            for u in range(0, len(equations)):
                if len(result[i]) > u:
                    result[i][u].extend(temp[i][u])
                else:
                    result[i].append(temp[i][u])

        for i in range(0, len(equations)):
            valuesLocal[i] = [temp[3][i][-1], None]
            valuesLocal[i][0] = valuesLocal[i][0] + eval(deltaLocal[i])

    return result


e = 0.001
startN = 2
maxN = 999999999
equationsStart = ["2*x + 2*z - y", "x + 2*z", "-2*x + y - z"]
# https://www.kontrolnaya-rabota.ru/s/equal-many/system-diff/?ef-TOTAL_FORMS=52&ef-INITIAL_FORMS=2&ef-MAX_NUM_FORMS=1000&ef-0-s=x%27+%3D+2x%2B2z-y&ef-1-s=y%27+%3D+x+%2B+2z&ef-2-s=z%27%3Dx-2y-z&ef-3-s=&ef-4-s=&ef-5-s=&ef-6-s=&ef-7-s=&ef-8-s=&ef-9-s=&ef-10-s=&ef-11-s=&ef-12-s=&ef-13-s=&ef-14-s=&ef-15-s=&ef-16-s=&ef-17-s=&ef-18-s=&ef-19-s=&ef-20-s=&ef-21-s=&ef-22-s=&ef-23-s=&ef-24-s=&ef-25-s=&ef-26-s=&ef-27-s=&ef-28-s=&ef-29-s=&ef-30-s=&ef-31-s=&ef-32-s=&ef-33-s=&ef-34-s=&ef-35-s=&ef-36-s=&ef-37-s=&ef-38-s=&ef-39-s=&ef-40-s=&ef-41-s=&ef-42-s=&ef-43-s=&ef-44-s=&ef-45-s=&ef-46-s=&ef-47-s=&ef-48-s=&ef-49-s=&ef-50-s=&ef-51-s=
equationsExactStart = ["-6 * c[0] * exp(t) + c[1]*(sin(t) - cos(t)) + c[2]*(-sin(t) - cos(t))",
                       "-4 * c[0] * exp(t) + c[1]*(sin(t) - cos(t)) + c[2]*(-sin(t) - cos(t))",
                       "c[0] * exp(t) + c[1]*cos(t) + c[2]*sin(t)"]
intervalStart = [1, 4]
valuesStart = [[2, None], [2, None], [-1, None]]
variablesStart = ["t", ["x", "y", "z"]]
c = [2, 2, -1]

equationsFindC = ["x / exp(t)", "y / exp(2 * t) + c[0] * exp(-t)",
                  "z / exp(2 * t) - c[1] * t"]
equationsDelta = ["0.07 * j * x + 0.6 + 0.3", "0.1*x + 0.05*j", "0.06"]

imp = solveImpulsesCauchySystem(equationsStart, variablesStart, valuesStart, intervalStart, e, startN, maxN,
                                equationsExactStart, equationsFindC, 5, equationsDelta)


table = np.array([imp[3][0]])
 
columns = copy.deepcopy(variablesStart[1])
for index in range(0, len(equationsStart)):
    if index != 0:
        table = np.append(table, [imp[3][index]], axis=0)
    table = np.append(table, [imp[4][index]], axis=0)
    columns.append(variablesStart[1][index] + "-exact")

    print(len(imp[2]));

    plt.plot(imp[2], imp[3][index])

plt.show()

# df = pd.DataFrame(table.transpose(), imp[2], columns)
# df.to_csv('result.csv', '\t', 'utf-8')
