# y'i+1 = yi +h*f(ti,yi)
# yi+1 = yi +h*(f(ti,yi) + f(ti+1,y'i+1))/2

import math
import matplotlib.pyplot as plt


def function(x, y):
    return 1 -(1-2*x)*y/(x*x);


def delta(y, j):
    return 0.003 * y * j -2


def solve(value=None, interval=None, epsilon=0.01):
    if value is None:
        value = [1, None]
    if interval is None:
        interval = [0, 10]

    n = 2

    h = (interval[1] - interval[0]) / n
    x_prev = [interval[0]]
    y_prev = [value[0]]
    x_current = [interval[0]]
    y_current = [value[0]]
    sign = 1

    index = 0
    while index != n / 2 + 1:
        x_prev = x_current
        y_prev = y_current
        x_current = [x_current[0]]
        y_current = [y_current[0]]

        n *= 2
        h = (interval[1] - interval[0]) / n

        for i in range(0, n):
            x_current.append(x_current[i] + sign * h)
            y_current_local = y_current[i] + sign * h * function(
                x_current[i], y_current[i]
            )
            y_current.append(
                y_current[i]
                + sign
                * h
                * (
                    function(x_current[i], y_current[i])
                    + function(x_current[i + 1], y_current_local)
                )
                / 2
            )

        index = 0
        while (
            index < len(x_prev) and abs(y_prev[index] - y_current[index * 2]) < epsilon
        ):
            index += 1

    return n / 2, h * 2, x_prev, y_prev


def solveImpulses(value=None, interval=None, e=0.01, impulses=3):
    if value is None:
        value = [1, None]
    if interval is None:
        interval = [0, 10]
    result = {}
    intervalLocal = [value[0], value[0]]
    valuesLocal = [value[0], None]

    for j in range(0, impulses + 1):
        intervalLocal[0] = intervalLocal[1]
        intervalLocal[1] = interval[0] + (interval[1] - interval[0]) / (
            impulses + 1
        ) * (j + 1)

        current = solve(valuesLocal, intervalLocal, e)

        if result.get("x") is not None:
            result.get("x").extend(current[2])
        else:
            result["x"] = current[2]
        if result.get("y") is not None:
            result.get("y").extend(current[3])
        else:
            result["y"] = current[3]
        valuesLocal[0] = result.get("y")[-1] + delta(result.get("y")[-1], j)

    return result


e = 0.001
interval = [1, 1.8]
values = [1, None]
impulses = 3

result = solve(values, interval, e)

impulsesResult = solveImpulses(values, interval, e, impulses)
for i in result[2]:
    print(i);
plt.plot(result[2], result[3], "r--")
plt.plot(impulsesResult.get("x"), impulsesResult.get("y"))
plt.show()
