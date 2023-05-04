import math
import numpy as np

import settings


def gen_fib(min_fib: int) -> list[int]:
    fib_seq = [1, 1]
    while fib_seq[-1] <= min_fib:
        fib_seq.append(fib_seq[-1] + fib_seq[-2])
    return fib_seq


def fibonacci(eps, func, a, b):
    count_iter, count_func, prev_len = 1, 2, b - a
    arr_a, arr_b, arr_ratio = [a], [b], []
    fib_seq = gen_fib(math.ceil((b - a) / eps))
    n = len(fib_seq) - 1
    if n < 2:
        x1, x2 = a, b
    else:
        x1, x2 = a + fib_seq[n - 2] / fib_seq[n] * (b - a), a + fib_seq[n - 1] / fib_seq[n] * (b - a)
    f1, f2 = func(x1), func(x2)
    k = 1
    while k + 2 < n and settings.is_work_correct(arr_ratio):
        if f1 > f2:
            a = x1
            if n - k < 2:
                x1 = a
            else:
                x1 = a + fib_seq[n - k - 2] / fib_seq[n - k] * (b - a)
            x2 = a + fib_seq[n - k - 1] / fib_seq[n - k] * (b - a)
            f1, f2 = f2, func(x2)
        else:
            b = x2
            x1 = a + fib_seq[n - k - 2] / fib_seq[n - k] * (b - a)
            if n - k < 1:
                x2 = b
            else:
                x2 = a + fib_seq[n - k - 1] / fib_seq[n - k] * (b - a)
            f1, f2 = func(x1), f1
        arr_a.append(a)
        arr_b.append(b)
        count_iter += 1
        count_func += 1
        k += 1
        arr_ratio.append(prev_len / (b - a))
        prev_len = b - a
    return (a + b) / 2, count_iter, count_func, arr_a, arr_b, arr_ratio


print("\n===============Fibonacci===============")
# Tests
val = fibonacci(0.001, settings.f, settings.A, settings.B)[0]
print(f"Алгоритм нашел минимум функции в точке {val}")

val = fibonacci(0.005, np.sin, -2.5, 2)[0]
print(f"Алгоритм нашел минимум функции sin(x) в точке {val}")

val = fibonacci(0.005, np.cos, -2.5, 2)[0]
print(f"Алгоритм нашел минимум функции cos(x) в точке {val}")
