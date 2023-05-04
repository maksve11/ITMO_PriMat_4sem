import numpy as np

import settings


def dichotomy(eps, func, a, b):
    count_iter, count_func, prev_len = 0, 0, b - a
    arr_a, arr_b, arr_ratio = [a], [b], []

    while b - a > eps and settings.is_work_correct(arr_ratio):
        count_iter += 1

        x1, x2 = (a + b) / 2 - eps / 2, (a + b) / 2 + eps / 2
        f_x1, f_x2 = func(x1), func(x2)
        count_func += 2

        if f_x1 < f_x2:
            b = x2
        elif f_x1 > f_x2:
            a = x1
        else:
            a, b = x1, x2

        arr_a.append(a)
        arr_b.append(b)
        arr_ratio.append(prev_len / (b - a))
        prev_len = b - a

    return (a + b) / 2, count_iter, count_func, arr_a, arr_b, arr_ratio


# Входные аргументы: Повторяются во всех методах
#
# eps: требуемая точность решения
# delta: параметр метода (расстояние между точками в новом отрезке), только в dichotomy
# func: целевая функция
# a: левая граница отрезка
# b: правая граница отрезка
#
# Выходные значения:
#
# (a + b) / 2: значение минимума функции
# count_iter: количество выполненных итераций метода
# count_func: количество вызовов функции func
# arr_a: массив левых границ отрезков на каждой итерации
# arr_b: массив правых границ отрезков на каждой итерации
# arr_ratio: массив отношений длин текущего и предыдущего отрезков на каждой итерации.


print("\n===============Dichotomy===============")
# Tests
val = dichotomy(0.001, settings.f, settings.A, settings.B)[0]
print(f"Алгоритм нашел минимум функции sin(0.5*ln(x)*x)+1 в точке {val}")

val = dichotomy(0.005, np.sin, -2.5, 2)[0]
print(f"Алгоритм нашел минимум функции sin(x) в точке {val}")

val = dichotomy(0.005, np.cos, -2.5, 2)[0]
print(f"Алгоритм нашел минимум функции cos(x) в точке {val}")
