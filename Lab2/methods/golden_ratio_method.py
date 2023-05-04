import numpy as np

import settings


def golden_ratio(eps, func, a, b):
    golden_const_1 = ((-5 ** 0.5 + 3) / 2)
    golden_const_2 = 1 / ((5 ** 0.5 + 1) / 2)
    count_iter, count_func, prev_len = 0, 0, b - a
    arr_a, arr_b, arr_ratio = [a], [b], []
    saved_part = 0
    prev_func = None
    while b - a > eps and settings.is_work_correct(arr_ratio):
        x1, x2 = a + golden_const_1 * (b - a), a + golden_const_2 * (b - a)
        if saved_part == 0:
            f_x1, f_x2 = func(x1), func(x2)
            count_func += 2
        elif saved_part == -1:
            f_x1, f_x2 = prev_func, func(x2)
            count_func += 1
        else:
            f_x1, f_x2 = func(x1), prev_func
            count_func += 1
        if f_x1 < f_x2:
            b = x2
            saved_part = 1
            prev_func = f_x1
        elif f_x1 > f_x2:
            a = x1
            saved_part = -1
            prev_func = f_x2
        else:
            a, b = x1, x2
            saved_part = 0
        arr_ratio.append(prev_len / (b - a))
        prev_len = b - a
        count_iter += 1
        arr_a.append(a), arr_b.append(b)

    return (a + b) / 2, count_iter, count_func, arr_a, arr_b, arr_ratio


print("\n===============Golden_ratio===============")
# Tests
val = golden_ratio(0.001, settings.f, settings.A, settings.B)[0]
print(f"Алгоритм нашел минимум функции в точке {val}")

val = golden_ratio(0.005, np.sin, -2.5, 2)[0]
print(f"Алгоритм нашел минимум функции sin(x) в точке {val}")

val = golden_ratio(0.005, np.cos, -2.5, 2)[0]
print(f"Алгоритм нашел минимум функции cos(x) в точке {val}")
