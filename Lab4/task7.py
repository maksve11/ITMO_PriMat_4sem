import numpy as np
from scipy.linalg import hilbert
import time
import matplotlib.pyplot as plt

from main import generate_rhs


def solve_gauss(A, b):
    n = A.shape[0]
    Ab = np.hstack((A, b[:, np.newaxis]))

    for i in range(n):
        max_row = np.argmax(np.abs(Ab[i:, i])) + i

        # Свапнуть строки
        Ab[[i, max_row]] = Ab[[max_row, i]]

        # Элиминация
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j] -= factor * Ab[i]

    x = np.zeros(n)

    # Обратная подстановка
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, :-1], x)) / Ab[i, i]

    return x


def seidel(A, b, x0, max_iterations=1000, tolerance=1e-6):
    n = A.shape[0]
    x = x0.copy()
    residual = np.inf
    iterations = 0

    while residual > tolerance and iterations < max_iterations:
        x_new = np.zeros_like(x)

        for i in range(n):
            x_new[i] = (b[i] - np.dot(A[i, :i], x_new[:i]) - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

        residual = np.linalg.norm(x_new - x)
        x = x_new
        iterations += 1

    return x, iterations, residual


# Параметры для сравнения
n_values = [10, 50, 100, 1000, 10000]

# Замер времени выполнения прямого метода (Гаусса)
gauss_times = []

for n in n_values:
    A = hilbert(n)
    F = generate_rhs(n)

    start_time = time.time()
    x_gauss = solve_gauss(A, F)
    gauss_time = time.time() - start_time
    gauss_times.append(gauss_time)

# Замер времени выполнения итерационного метода (Зейделя)
seidel_times = []

for n in n_values:
    A = hilbert(n)
    F = generate_rhs(n)
    x0 = np.zeros(n)

    start_time = time.time()
    x_seidel, _, _ = seidel(A, F, x0)
    seidel_time = time.time() - start_time
    seidel_times.append(seidel_time)

# Вывод результатов
print("=== Прямой метод (Гаусс) ===")
print("Гаусс times:", gauss_times)
print("=== Итерационный метод (Зейдель) ===")
print("Seidel times:", seidel_times)

# Построение графиков
plt.plot(n_values, gauss_times, marker="o", label="Gauss Times")
plt.plot(n_values, seidel_times, marker="o", label="Seidel Times")
plt.title("Execution Times for Different Matrix Sizes")
plt.xlabel("Matrix Size (n)")
plt.ylabel("Execution Time (s)")
plt.legend()
plt.show()