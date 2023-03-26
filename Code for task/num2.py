import numpy as np
from matplotlib import pyplot as plt


def get_rightdiff_atpoint(function, argument, h):
    return (function(argument + h) - function(argument)) / h


def get_rightdiff(function, arguments, h):
    return [get_rightdiff_atpoint(function, x, h) for x in arguments]


def get_leftdiff_atpoint(function, argument, h):
    return (function(argument) - function(argument - h)) / h


def get_leftdiff(function, arguments, h):
    return [get_leftdiff_atpoint(function, x, h) for x in arguments]


def get_diff_atpoint(function, argument, h):
    return (function(argument + h) - function(argument - h)) / (2 * h)


def get_diff(function, arguments, h):
    return [get_diff_atpoint(function, x, h) for x in arguments]


a, b = 10, 40
h = 2e-1
argx = np.linspace(a, b, int((b - a) / h) + 1)


def func_first(x):
    return x ** 2 * np.e ** np.sin(x)


def func_second(x):
    return np.log(x) * np.e ** np.sin(x)


fig, ax = plt.subplots(1, 2, figsize=(20, 8))
ax[0].plot(argx, func_first(argx))
ax[1].plot(argx, func_second(argx))
ax[0].grid()
ax[1].grid()
ax[0].set(title=f'Graph of first initiall function', xlabel='$x$', ylabel="$f_1(x)$")
ax[1].set(title=f'Graph of second initiall function', xlabel='$x$', ylabel="$f_2(x)$")
plt.show()


def func_derivative_first(x):
    return np.e ** np.sin(x) * (2 * x + x ** 2 * np.cos(x))


def func_derivative_second(x):
    return np.e ** np.sin(x) * (1 / x + np.log(x) * np.cos(x))


analyticdiff_first = func_derivative_first(argx)
analyticdiff_second = func_derivative_second(argx)

# first function
rightdiff_first = get_rightdiff(func_first, argx, h)
leftdiff_first = get_leftdiff(func_first, argx, h)
diff_first = get_diff(func_first, argx, h)

# second function
rightdiff_second = get_rightdiff(func_second, argx, h)
leftdiff_second = get_leftdiff(func_second, argx, h)
diff_second = get_diff(func_second, argx, h)


def plot_compare(func_i : str, derivpairs, arguments, h):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    for i in range(len(derivpairs)):
        ax.scatter(arguments, derivpairs[i][0], label=f"{derivpairs[i][1]} method", s=50)
    ax.grid()
    ax.legend()
    ax.set(title=f'Comparison of derivative values with step $h={h}$ for {func_i} function',
           xlabel='$x$', ylabel="$y'(x)$")
    plt.show()


plot_compare("first", [[rightdiff_first[-3:], "right"],
                       [leftdiff_first[-3:], "left"],
                       [diff_first[-3:], "center"],
                       [analyticdiff_first[-3:], "analytic"]], argx[-3:], h)
plot_compare("second", [[rightdiff_second[-3:], "right"],
                        [leftdiff_second[-3:], "left"],
                        [diff_second[-3:], "center"],
                        [analyticdiff_second[-3:], "analytic"]], argx[-3:], h)