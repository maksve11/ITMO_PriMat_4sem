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


MSE_left_first = sum((x - a) ** 2 for x, a in zip(leftdiff_first, analyticdiff_first)) / len(analyticdiff_first)
MSE_right_first = sum((x - a) ** 2 for x, a in zip(rightdiff_first, analyticdiff_first)) / len(analyticdiff_first)
MSE_center_first = sum((x - a) ** 2 for x, a in zip(diff_first, analyticdiff_first)) / len(analyticdiff_first)

MSE_left_second = sum((x - a) ** 2 for x, a in zip(leftdiff_second, analyticdiff_second)) / len(analyticdiff_second)
MSE_right_second = sum((x - a) ** 2 for x, a in zip(rightdiff_second, analyticdiff_second)) / len(analyticdiff_second)
MSE_center_second = sum((x - a) ** 2 for x, a in zip(diff_second, analyticdiff_second)) / len(analyticdiff_second)



def get_MSEs(n, func):
    argx_n = np.linspace(a, b, n * int((b - a) / h) + 1)
    analyticdiff_first_n = func_derivative_first(argx_n)
    analyticdiff_second_n = func_derivative_second(argx_n)
    MSE_right_first_n = sum((x - a) ** 2 for x, a in zip(get_rightdiff(func, argx_n, h / n), analyticdiff_first_n)) / len(analyticdiff_first_n)
    MSE_center_first_n = sum((x - a) ** 2 for x, a in zip(get_diff(func, argx_n, h / n), analyticdiff_first_n)) / len(analyticdiff_first_n)
    return [MSE_right_first_n, MSE_center_first_n]


# first function, h / 2
MSE_right_first_2, MSE_center_first_2 = get_MSEs(2, func_first)
# first function, h / 4
MSE_right_first_4, MSE_center_first_4 = get_MSEs(4, func_first)
# first function, h / 8
MSE_right_first_8, MSE_center_first_8 = get_MSEs(8, func_first)
# first function, h / 16
MSE_right_first_16, MSE_center_first_16 = get_MSEs(16, func_first)


steps = [h / 2 ** i for i in range(5)]
fig, ax = plt.subplots(1, 2, figsize=(20, 8))
ax[0].plot(steps,
           [MSE_right_first, MSE_right_first_2, MSE_right_first_4, MSE_right_first_8, MSE_right_first_16],
           'o-', label="right method",
           markersize=8)
ax[1].plot(steps,
           [MSE_center_first, MSE_center_first_2, MSE_center_first_4, MSE_center_first_8, MSE_center_first_16],
           'o-', label="center method",
           markersize=8)
ax[0].grid()
ax[1].grid()
ax[0].legend()
ax[1].legend()
ax[0].set(xlabel="step $h$", ylabel="$MSE_{right}$", title="$MSE_{right}$ graph for first function", xticks=steps)
ax[1].set(xlabel="step $h$", ylabel="$MSE_{center}$", title="$MSE_{center}$ graph for first function", xticks=steps)
plt.show()