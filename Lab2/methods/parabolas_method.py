import numpy as np
import settings


def parabolas(eps, func, a, b):
    count_iter, count_func, prev_len = 0, 3, b - a
    x1, x2, x3 = a, (b + a) / 2, b
    arr_a, arr_b, arr_ratio = [x1], [x3], []
    f1, f2, f3 = func(x1), func(x2), func(x3)
    while True:
        numerator = ((x2 - x1) ** 2 * (f2 - f3) - (x2 - x3) ** 2 * (f2 - f1))
        denominator = 2 * ((x2 - x1) * (f2 - f3) - (x2 - x3) * (f2 - f1))
        if denominator == 0 or abs(x3 - x1) < eps or not settings.is_work_correct(arr_ratio):
            break
        x_min = x2 - numerator / denominator
        f_min = func(x_min)

        if x_min < x2:
            if f_min > f2:
                x1 = x_min
                f1 = f_min
            else:
                x3, x2 = x2, x_min
                f3, f2 = f2, f_min
        else:
            if f_min > f2:
                x3 = x_min
                f3 = f_min
            else:
                x1, x2 = x2, x_min
                f1, f2 = f2, f_min

        count_iter += 1
        count_func += 1
        arr_a.append(x1)
        arr_b.append(x3)
        arr_ratio.append(prev_len / (x3 - x1))
        prev_len = x3 - x1
    if x2 == x1 or f2 == f1:
        res = arr_a[-1], count_iter, count_func, arr_a, arr_b, arr_ratio
    else:
        res = arr_b[-1], count_iter, count_func, arr_a, arr_b, arr_ratio
    return res


print("\n===============Parabolas===============")
# Tests
val = parabolas(0.001, settings.f, settings.A, settings.B)[0]
print(f"Алгоритм нашел минимум функции в точке {val}")

val = parabolas(0.005, np.sin, -2.5, 2)[0]
print(f"Алгоритм нашел минимум функции sin(x) в точке {val}")

val = parabolas(0.005, np.cos, -2.5, 2)[0]
print(f"Алгоритм нашел минимум функции cos(x) в точке {val}")
