import numpy as np
from matplotlib import pyplot as plt


def get_leftsum(f, arguments, h):
    return sum(h * f(arg) for arg in arguments[:-1])


def get_rightsum(f, arguments, h):
    return sum(h * f(arg) for arg in arguments[1:])


def get_centersum(f, arguments, h):
    return sum(h * f((arguments[i] + arguments[i + 1]) / 2) for i in range(len(arguments) - 1))


def get_trapezoidsum(f, arguments, h):
    argf = [f(arg) for arg in arguments]
    return sum(h * (argf[i] + argf[i + 1]) / 2 for i in range(len(argf) - 1))


def get_parabolicsum(f, arguments, h):
    return sum(h * (f(arguments[i]) + f(arguments[i + 1]) + 4 * f((arguments[i] + arguments[i + 1]) / 2)) / 6 for i in range(len(arguments) - 1))


def function_1(x):
    return np.cos(x / 10) + 2


def function_2(x):
    return np.sin(x) * np.e ** np.cos(x) + 10


a, b = 10, 40


def sum_analytics_first(a, b):
    return (2 * b + 10 * np.sin(b / 10) - 2 * a - 10 * np.sin(a / 10))


def sum_analytics_second(a, b):
    return (10 * b - np.e ** np.cos(b) - 10 * a + np.e ** np.cos(a))


h = 2e-1
argx = np.linspace(a, b, int((b - a) / h) + 1)


def get_deviations(lst_n, sum_method, function, value_analytics):
    return [abs(sum_method(function, np.linspace(a, b, n * int((b - a) / h) + 1), h / n) - value_analytics(a, b)) for n in lst_n]


ns = [2, 4, 8, 16]
deviations = [[[get_deviations(ns, get_centersum, function_1, sum_analytics_first), "center sums"],
               [get_deviations(ns, get_trapezoidsum, function_1, sum_analytics_first), "trapezoid sums"],
               [get_deviations(ns, get_parabolicsum, function_1, sum_analytics_first), "parabolic sums"]],
              [[get_deviations(ns, get_centersum, function_2, sum_analytics_second), "center sums"],
               [get_deviations(ns, get_trapezoidsum, function_2, sum_analytics_second), "trapezoid sums"],
               [get_deviations(ns, get_parabolicsum, function_2, sum_analytics_second), "parabolic sums"]]]


steps = [h / 2 ** i for i in range(4)]
fig, ax = plt.subplots(1, len(deviations), figsize=(20, 4))
for dev_i in range(len(deviations)):
    for method in deviations[dev_i]:
        ax[dev_i].plot(steps, method[0], 'o-', label=method[1])
    ax[dev_i].grid()
    ax[dev_i].legend()
    ax[dev_i].set(xlabel="step $h$", ylabel="deviation",
                  title=f"Deviation graph for function {dev_i + 1}",
                  xticks=steps)
plt.show()