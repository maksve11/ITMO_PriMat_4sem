import numpy as np


def seidel(A, b, x0, tol=1e-6, max_iter=100):
    n = len(A)
    x = x0.copy()
    iterations = 0
    residual = np.linalg.norm(A @ x - b)

    while residual > tol and iterations < max_iter:
        for i in range(n):
            x[i] = (b[i] - A[i, :i] @ x[:i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]

        iterations += 1
        residual = np.linalg.norm(A @ x - b)

    return x, iterations, residual


# Пример использования
A = np.array([[4, 1, -1],
              [2, 7, 1],
              [1, -3, 12]])
b = np.array([3, 19, 31])
x0 = np.zeros_like(b)

x, iterations, residual = seidel(A, b, x0)

print("Solution:")
print(x)
print("Number of iterations:", iterations)
print("Residual:", residual)