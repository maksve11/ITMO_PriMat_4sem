import numpy as np


def gaussian_elimination(A, b):
    n = len(A)

    # Прямой ход
    for i in range(n):
        # Выбор ведущего элемента
        max_index = i
        for j in range(i + 1, n):
            if abs(A[j][i]) > abs(A[max_index][i]):
                max_index = j
        A[[i, max_index]] = A[[max_index, i]]
        b[[i, max_index]] = b[[max_index, i]]

        # Приведение матрицы к треугольному виду
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            A[j] -= factor * A[i]
            b[j] -= factor * b[i]

    # Обратный ход
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i][i + 1:], x[i + 1:])) / A[i][i]

    return x


# Пример использования
A = np.array([[2, -1, 3],
              [1, 3, -2],
              [4, 1, -1]], dtype=float)

b = np.array([10, 5, 12], dtype=float)

x = gaussian_elimination(A, b)
print(x)