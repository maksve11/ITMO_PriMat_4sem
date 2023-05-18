import numpy as np


def line_search_golden_section(function, point, direction, initial_step_size, epsilon):
    """
    Метод золотого сечения для одномерной оптимизации.

    Аргументы:
    - function: функция, для которой выполняется одномерная оптимизация
    - point: текущая точка (x, y) для одномерной оптимизации
    - direction: направление для одномерной оптимизации
    - initial_step_size: начальный шаг для одномерной оптимизации
    - epsilon: точность для определения условия остановки

    Возвращает:
    - step_size: оптимальный шаг, найденный методом золотого сечения
    """

    a = 0
    b = initial_step_size
    tau = (np.sqrt(5) - 1) / 2  # Значение золотого сечения

    # Функция для вычисления значения функции в заданной точке
    def f(alpha):
        return function(*(point + alpha * direction))

    # Начальные значения
    alpha1 = b - tau * (b - a)
    alpha2 = a + tau * (b - a)
    f1 = f(alpha1)
    f2 = f(alpha2)

    while b - a > epsilon:
        if f1 < f2:
            b = alpha2
            alpha2 = alpha1
            alpha1 = b - tau * (b - a)
            f2 = f1
            f1 = f(alpha1)
        else:
            a = alpha1
            alpha1 = alpha2
            alpha2 = a + tau * (b - a)
            f1 = f2
            f2 = f(alpha2)

    step_size = (a + b) / 2  # Оптимальный шаг

    return step_size