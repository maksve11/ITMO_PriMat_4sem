import numpy as np


def generate_quadratic_function(n, k):
    A = np.random.rand(n, n)  # Генерируем случайную матрицу размером (n, n)
    A = (A + A.T) / 2  # Делаем матрицу симметричной
    eigvals = np.linalg.eigvalsh(A)  # Вычисляем собственные значения матрицы

    # Масштабируем собственные значения для получения заданного числа обусловленности
    scaled_eigvals = (k / np.max(eigvals)) * eigvals

    # Создаем квадратичную функцию с матрицей A и собственными значениями scaled_eigvals
    def quadratic_function(x):
        return 0.5 * np.dot(x.T, np.dot(A, x))

    return quadratic_function


# Пример использования
n = 3  # Количество переменных
k = 10  # Число обусловленности
quadratic_func = generate_quadratic_function(n, k)

# Генерируем случайную точку
x = np.random.rand(n)

# Вычисляем значение квадратичной функции в точке x
result = quadratic_func(x)
print("Result:", result)