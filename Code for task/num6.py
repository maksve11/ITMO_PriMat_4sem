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

# first function
analytics_sum_first = sum_analytics_first(a, b)
left_sum_first = get_leftsum(function_1, argx, h)
right_sum_first = get_rightsum(function_1, argx, h)
center_sum_first = get_centersum(function_1, argx, h)
trapezoid_sum_first = get_trapezoidsum(function_1, argx, h)
parabolic_sum_first = get_parabolicsum(function_1, argx, h)
sums_first = [[analytics_sum_first, "absolute sums"],
              [center_sum_first, "center sums"],
              [trapezoid_sum_first, "trapezoid sums"],
              [parabolic_sum_first, "parabolic sums"]]

# second function
analytics_sum_second = sum_analytics_second(a, b)
left_sum_second = get_leftsum(function_2, argx, h)
right_sum_second = get_rightsum(function_2, argx, h)
center_sum_second = get_centersum(function_2, argx, h)
trapezoid_sum_second = get_trapezoidsum(function_2, argx, h)
parabolic_sum_second = get_parabolicsum(function_2, argx, h)
sums_second = [[analytics_sum_second, "absolute sums"],
              [center_sum_second, "center sums"],
              [trapezoid_sum_second, "trapezoid sums"],
              [parabolic_sum_second, "parabolic sums"]]

print(f"For first function:", *[f"{entry[1]}: {entry[0]}" for entry in sums_first], sep="\n")
print(f"For second function:", *[f"{entry[1]}: {entry[0]}" for entry in sums_second], sep="\n")

fig, ax = plt.subplots(1, 2, figsize=(20, 4))
for i in range(len(sums_first)):
    ax[0].plot(i, sums_first[i][0], 'o', label=sums_first[i][1], markersize=10)
ax[0].grid()
ax[0].legend()
ax[0].set(xticks=np.arange(0, len(sums_first)), xlabel="summation method id", ylabel="summation value", title="Comparison of integration results for first function")
for i in range(len(sums_second)):
    ax[1].plot(i, sums_second[i][0], 'o', label=sums_second[i][1], markersize=10)
ax[1].grid()
ax[1].legend()
ax[1].set(xticks=np.arange(0, len(sums_second)), xlabel="summation method id", ylabel="summation value", title="Comparison of integration results for second function")
plt.show()
