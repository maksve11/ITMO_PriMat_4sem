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
print(f"First function\nleft derivatives MSE: {MSE_left_first}\nright derivatives MSE: {MSE_right_first}\ncenter derivatives MSE: {MSE_center_first}")
MSE_left_second = sum((x - a) ** 2 for x, a in zip(leftdiff_second, analyticdiff_second)) / len(analyticdiff_second)
MSE_right_second = sum((x - a) ** 2 for x, a in zip(rightdiff_second, analyticdiff_second)) / len(analyticdiff_second)
MSE_center_second = sum((x - a) ** 2 for x, a in zip(diff_second, analyticdiff_second)) / len(analyticdiff_second)
print(f"Second function\nleft derivatives MSE: {MSE_left_second}\nright derivatives MSE: {MSE_right_second}\ncenter derivatives MSE: {MSE_center_second}")