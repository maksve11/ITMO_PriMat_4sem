import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


def lu_decomposition_sparse(A):
    n = A.shape[0]
    L = csr_matrix((n, n), dtype=float)
    U = csr_matrix((n, n), dtype=float)

    for k in range(n):
        L[k, k] = 1.0
        U[k, k] = A[k, k] - L[k, :k].dot(U[:k, k].toarray().flatten())

        for j in range(k + 1, n):
            U[k, j] = A[k, j] - L[k, :k].dot(U[:k, j].toarray().flatten())

        for i in range(k + 1, n):
            L[i, k] = (A[i, k] - L[i, :k].dot(U[:k, k].toarray().flatten())) / U[k, k]

    return L, U


# Пример использования
A = np.array([[2, -1, 0],
              [-1, 2, -1],
              [0, -1, 2]])

# Создание разреженной матрицы в формате CSR
A_sparse = csr_matrix(A)

L, U = lu_decomposition_sparse(A_sparse)

print("L:")
print(L.toarray())
print("U:")
print(U.toarray())

# Решение СЛАУ с использованием LU-разложения
b = np.array([1, 0, -1])

# Решение СЛАУ
y = spsolve(L, b)
x = spsolve(U, y)

print("Solution:")
print(x)