import numpy as np
from numpy.linalg import cond
import matplotlib.pyplot as plt

from main import generate_matrix, generate_rhs, exact_solution, lu_decomposition, solve_lu
from methods.seidel import seidel


def evaluate_methods(n, k):
    A_gauss = generate_matrix(n, k)
    A_lu = generate_matrix(n, k)
    A_seidel = generate_matrix(n, k)
    F = generate_rhs(n)
    x_exact = exact_solution(n)

    print("Matrix A^(k):")
    print(A_gauss)
    print("Condition number:", cond(A_gauss))

    # Метод Гаусса с выбором ведущего элемента
    x_gauss = np.linalg.solve(A_gauss, F)
    gauss_error = np.linalg.norm(x_gauss - x_exact)
    print("Gauss error:", gauss_error)

    # Метод LU-разложения
    L, U = lu_decomposition(A_lu)
    x_lu = solve_lu(L, U, F)
    lu_error = np.linalg.norm(x_lu - x_exact)
    print("LU error:", lu_error)

    # Метод Зейделя
    x0 = np.zeros(n)
    x_seidel, iterations, residual = seidel(A_seidel, F, x0)
    seidel_error = np.linalg.norm(x_seidel - x_exact)
    print("Seidel error:", seidel_error)

    return cond(A_gauss), gauss_error, lu_error, seidel_error


# Параметры для оценки
n = 5  # Размерность матрицы
k_values = [1, 2, 3, 4, 5]  # Значения k для изменения обусловленности

condition_numbers = []
gauss_errors = []
lu_errors = []
seidel_errors = []

# Оценка для каждого значения k
for k in k_values:
    print("==== k =", k, "====")
    cond_number, gauss_error, lu_error, seidel_error = evaluate_methods(n, k)
    condition_numbers.append(cond_number)
    gauss_errors.append(gauss_error)
    lu_errors.append(lu_error)
    seidel_errors.append(seidel_error)
    print()

# Вывод результатов
print("Condition numbers:", condition_numbers)
print("Gauss errors:", gauss_errors)
print("LU errors:", lu_errors)
print("Seidel errors:", seidel_errors)

# Построение графиков
plt.plot(condition_numbers, gauss_errors, marker="o")
plt.title("Gauss Error")
plt.xlabel("Condition Number")
plt.ylabel("Error")
plt.show()

plt.plot(condition_numbers, lu_errors, marker="o")
plt.title("LU Error")
plt.xlabel("Condition Number")
plt.ylabel("Error")
plt.show()

plt.plot(condition_numbers, seidel_errors, marker="o")
plt.title("Seidel Error")
plt.xlabel("Condition Number")
plt.ylabel("Error")
plt.show()