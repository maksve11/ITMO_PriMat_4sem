import numpy as np
from numpy.linalg import cond


def generate_matrix(n, k):
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                A[i, j] = np.random.choice([-4, -3, -2, -1])

        if i > 0:
            A[i, i] = -np.sum(A[i, :])
        else:
            A[i, i] = -np.sum(A[i, :]) + 10 ** (-k)

    return A


def generate_rhs(n):
    return np.arange(1, n + 1)


def exact_solution(n):
    return np.ones(n)


def lu_decomposition(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for k in range(n):
        L[k, k] = 1.0
        U[k, k] = A[k, k] - L[k, :k].dot(U[:k, k])

        for j in range(k + 1, n):
            U[k, j] = A[k, j] - L[k, :k].dot(U[:k, j])

        for i in range(k + 1, n):
            L[i, k] = (A[i, k] - L[i, :k].dot(U[:k, k])) / U[k, k]

    return L, U


def solve_lu(L, U, b):
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(U, y)
    return x