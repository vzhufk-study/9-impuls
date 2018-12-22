import copy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def function(equation, t, results, variables):
    equationLocal = equation.replace(variables[0], 't')
    i = 0
    for variable in variables[1]:
        equationLocal = equationLocal.replace(variable, 'results[' + str(i) + ']')
        i += 1
    return eval(equationLocal)


def solveCauchySystem(equations=None, variables=None, values=None, interval=None, e=0.01, N=2, maxN=9999999999,
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
    # if values[0] is None:
    #     x_old = [interval[1]]
    #     x_new = [interval[1]]
    #     y_old = [value[1]]
    #     result_new = [value[1]]
    #     flag = -1

    # print(result_new)
    # exit()
    for i in range(0, n):
        results = results_new
        results_new = []
        
        t_new.append(t_new[i] + flag * h)
        for j in range(0, size):
            temp_result = result_new[j][i] + flag * h * function(equations[j], t_new[i], results, variables)
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
                # print(result_new[j][i], h, function(equations[j], t_new[i], results, variables), temp_result)
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

    # if flag == -1:
    #     return n / 2, h * 2, x_old[::-1], y_old[::-1]
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

        temp = solveCauchySystem(equations, variables, valuesLocal, intervalLocal, e, N, maxN,
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


e = 0.01
# startN = 1200
startN = 2
# maxN = 2400
maxN = 999999999
equationsStart = ["x", "x+2*y", "x+y+2*z"]
equationsExactStart = ["c[0] * exp(t)", "c[1] * exp(2 * t) - c[0] * exp(t)",
                       "c[2] * exp(2 * t) + c[1] * t * exp(2 * t)"]
intervalStart = [0, 1.2]
# intervalStart = [0, 1]
valuesStart = [[5, None], [0, None], [4, None]]
variablesStart = ["t", ["x", "y", "z"]]
c = [5, 5, 4]

# first = solveCauchySystem(equationsStart, variablesStart, valuesStart, intervalStart, e, startN, maxN,
#                           equationsExactStart, c)
# print(first[0:2])
# out1 = [first[2][-1]]
# out2 = [first[2][-1]]
# for index in range(0, len(equationsStart)):
#     out1.append(first[3][index][-1])
#     out2.append(first[4][index][-1])
#
#     plt.plot(first[2], first[3][index])
#     plt.plot(first[2], first[4][index])
# plt.show()
# print(out1)
# print(out2)

equationsFindC = ["x / exp(t)", "y / exp(2 * t) + c[0] * exp(-t)",
                  "z / exp(2 * t) - c[1] * t"]
equationsDelta = ["0.3 * x", "y / 2", "j * x / 1000 + z / 3"]

imp = solveImpulsesCauchySystem(equationsStart, variablesStart, valuesStart, intervalStart, e, startN, maxN,
                                equationsExactStart, equationsFindC, 5, equationsDelta)

table = np.array([imp[3][0]])

columns = copy.deepcopy(variablesStart[1])
for index in range(0, len(equationsStart)):
    if index != 0:
        table = np.append(table, [imp[3][index]], axis=0)
    table = np.append(table, [imp[4][index]], axis=0)
    columns.append(variablesStart[1][index] + "-exact")

    plt.plot(imp[2], imp[3][index])
    plt.plot(imp[2], imp[4][index])

plt.show()

df = pd.DataFrame(table.transpose(), imp[2], columns)
df.to_csv('result.csv', '\t', 'utf-8')
