import numpy as np

from method import line_search_golden_section
from quadratic_funcs import f1, f1_gradient, f2, f2_gradient, f1_hessian, f2_hessian


def gradient_descent_constant_step(function, gradient_function, initial_point, learning_rate, num_iterations):
    """
    Градиентный спуск с постоянным шагом.

    Аргументы:
    - function: функция, для которой выполняется градиентный спуск
    - gradient_function: функция для вычисления градиента функции
    - initial_point: начальная точка (x0, y0) для градиентного спуска
    - learning_rate: постоянный шаг градиентного спуска
    - num_iterations: количество итераций градиентного спуска

    Возвращает:
    - optimal_point: оптимальная точка, достигнутая после градиентного спуска
    - optimal_value: оптимальное значение функции, достигнутое после градиентного спуска
    """

    point = np.array(initial_point)  # Преобразуем начальную точку в numpy массив

    for iteration in range(num_iterations):
        gradient = np.array(gradient_function(*point))  # Вычисляем градиент функции в текущей точке

        new_point = point - learning_rate * gradient  # Вычисляем новую точку с заданным шагом
        value = function(*new_point)  # Вычисляем значение функции в новой точке

        point = new_point  # Обновляем текущую точку

    return point, value


# Задание начальной точки, постоянного шага градиентного спуска и количества итераций
initial_point = [0, 0]
learning_rate = 0.1
num_iterations = 100

# Применение градиентного спуска с постоянным шагом к первой функции
optimal_point1, optimal_value1 = gradient_descent_constant_step(f2, f2_gradient, initial_point, learning_rate, num_iterations)
print("Optimal Point for f2:", optimal_point1)


def armijo_condition(function, point, gradient, step_size, c, tau):
    """
    Проверка условия Армихо для выбора подходящего шага градиентного спуска.

    Аргументы:
    - function: функция, для которой выполняется градиентный спуск
    - point: текущая точка (x, y) для проверки условия
    - gradient: градиент функции в текущей точке
    - step_size: текущий шаг градиентного спуска
    - c: параметр условия Армихо (0 < c < 1)
    - tau: коэффициент дробления шага (0 < tau < 1)

    Возвращает:
    - True, если условие Армихо выполняется
    - False, если условие Армихо не выполняется
    """

    new_point = point - step_size * gradient  # Вычисляем новую точку с заданным шагом
    function_value = function(*point)  # Значение функции в текущей точке
    new_function_value = function(*new_point)  # Значение функции в новой точке

    return new_function_value <= function_value - c * step_size * np.linalg.norm(gradient) ** 2


def gradient_descent_with_armijo(function, gradient_function, initial_point, initial_step_size, c, tau, max_iterations):
    """
    Метод градиентного спуска с дроблением шага и использованием условия Армихо.

    Аргументы:
    - function: функция, для которой выполняется градиентный спуск
    - initial_point: начальная точка (x0, y0) для градиентного спуска
    - initial_step_size: начальный шаг градиентного спуска
    - c: параметр условия Армихо (0 < c < 1)
    - tau: коэффициент дробления шага (0 < tau < 1)
    - max_iterations: максимальное количество итераций

    Возвращает:
    - optimal_point: оптимальная точка, достигнутая после градиентного спуска
    - optimal_value: оптимальное значение функции, достигнутое после градиентного спуска
    """

    def gradient(x, y):
        return np.array(gradient_function(x, y))

    point = np.array(initial_point)  # Преобразуем начальную точку в numpy массив
    step_size = initial_step_size  # Инициализируем шаг градиентного спуска
    optimal_point = point  # Инициализируем оптимальную точку значением начальной точки
    optimal_value = function(*point)  # Вычисляем значение функции в начальной точке
    iteration = 0  # Счет

    while iteration < max_iterations:
        gradient_value = gradient(*point)  # Вычисляем градиент функции в текущей точке

        if armijo_condition(function, point, gradient_value, step_size, c, tau):
            new_point = point - step_size * gradient_value  # Вычисляем новую точку с заданным шагом
            value = function(*new_point)  # Вычисляем значение функции в новой точке

            if value < optimal_value:
                optimal_point = new_point  # Обновляем оптимальную точку, если значение функции уменьшилось
                optimal_value = value

            point = new_point  # Обновляем текущую точку
            iteration += 1  # Увеличиваем счетчик итераций
        else:
            step_size *= tau  # Дробим шаг градиентного спуска

    return optimal_point, optimal_value


# Задание начальной точки, начального шага градиентного спуска, параметров Армихо и максимального количества итераций
initial_point = [0, 0]
initial_step_size = 0.1
c = 0.5
tau = 0.5
max_iterations = 100

# Применение градиентного спуска с дроблением шага и условием Армихо к первой функции
optimal_point1, optimal_value1 = gradient_descent_with_armijo(f1, f1_gradient, initial_point, initial_step_size, c, tau, max_iterations)
print("Optimal Point for f1:", optimal_point1)


def gradient_descent_with_line_search(function, gradient_function, initial_point, epsilon, max_iterations):
    """
    Метод наискорейшего спуска с использованием метода одномерной оптимизации.

    Аргументы:
    - function: функция, для которой выполняется градиентный спуск
    - initial_point: начальная точка (x0, y0) для градиентного спуска
    - epsilon: точность для определения условия остановки
    - max_iterations: максимальное количество итераций

    Возвращает:
    - optimal_point: оптимальная точка, достигнутая после градиентного спуска
    - optimal_value: оптимальное значение функции, достигнутое после градиентного спуска
    """

    point = np.array(initial_point)  # Преобразуем начальную точку в numpy массив
    optimal_point = point  # Инициализируем оптимальную точку значением
    optimal_value = function(*point)  # Вычисляем значение функции в начальной точке
    iteration = 0  # Счетчик итераций

    while iteration < max_iterations:
        gradient = np.array(gradient_function(*point))  # Вычисляем градиент функции в текущей точке
        direction = -gradient  # Направление спуска

        step_size = line_search_golden_section(function, point, direction, 1.0, epsilon)  # Определение шага

        new_point = point + step_size * direction  # Вычисляем новую точку с заданным шагом
        value = function(*new_point)  # Вычисляем значение функции в новой точке

        if value < optimal_value:
            optimal_point = new_point  # Обновляем оптимальную точку, если значение функции уменьшилось
            optimal_value = value

        point = new_point  # Обновляем текущую точку
        iteration += 1  # Увеличиваем счетчик итераций

    return optimal_point, optimal_value


# Задание начальной точки, точности и максимального количества итераций
initial_point = [0, 0]
epsilon = 1e-6
max_iterations = 1000

# Применение метода наискорейшего спуска с одномерной оптимизацией к первой функции
optimal_point1, optimal_value1 = gradient_descent_with_line_search(f1, f1_gradient, initial_point, epsilon, max_iterations)
print("Optimal Point for f1:", optimal_point1)


def conjugate_gradient_method(function, gradient_function, hessian_function, initial_point, epsilon, max_iterations):
    """
    Метод сопряженных градиентов для оптимизации квадратичной функции.

    Аргументы:
    - function: функция, для которой выполняется оптимизация
    - initial_point: начальная точка (x0, y0) для оптимизации
    - epsilon: точность для определения условия остановки
    - max_iterations: максимальное количество итераций

    Возвращает:
    - optimal_point: оптимальная точка, достигнутая после оптимизации
    - optimal_value: оптимальное значение функции, достигнутое после оптимизации
    """

    point = np.array(initial_point, dtype=np.float64)  # Преобразуем начальную точку в numpy массив
    gradient = np.array(gradient_function(*point))  # Вычисляем градиент функции в начальной точке
    direction = -gradient  # Направление спуска
    iteration = 0  # Счетчик итераций

    while iteration < max_iterations and np.linalg.norm(gradient) > epsilon:
        alpha = -(np.dot(gradient, direction) / np.dot(direction, hessian_function(*point)))  # Шаг
        point += alpha * direction  # Вычисляем градиент функции в новой точке

        new_gradient = np.array(gradient_function(*point))  # Вычисляем новую точку с заданным шагом
        beta = np.dot(new_gradient, new_gradient) / np.dot(gradient, gradient)  # Коэффициент

        direction = -new_gradient + beta * direction  # Обновляем направление спуска
        gradient = new_gradient
        iteration += 1  # Вычисляем значение функции в новой точке

    return point, function(*point)


# Задание начальной точки, точности и максимального количества итераций
initial_point = [0, 0]
epsilon = 1e-3
max_iterations = 1000

# Применение метода сопряженных градиентов к первой функции
optimal_point1, optimal_value1 = conjugate_gradient_method(f1, f1_gradient, f1_hessian, initial_point, epsilon, max_iterations)
print("Optimal Point for f1:", optimal_point1)