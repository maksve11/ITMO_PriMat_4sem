import numpy as np
from numpy.linalg import cond
from scipy.linalg import hilbert

from main import generate_rhs, exact_solution, lu_decomposition, solve_lu
from methods.seidel import seidel
import matplotlib.pyplot as plt


def evaluate_methods(n):
    A = hilbert(n)
    F = generate_rhs(n)
    x_exact = exact_solution(n)

    print("Matrix Hilbert(n):")
    print(A)
    print("Condition number:", cond(A))

    # Метод Гаусса с выбором ведущего элемента
    x_gauss = np.linalg.solve(A, F)
    gauss_error = np.linalg.norm(x_gauss - x_exact)
    print("Gauss error:", gauss_error)

    # Метод LU-разложения
    L, U = lu_decomposition(A)
    x_lu = solve_lu(L, U, F)
    lu_error = np.linalg.norm(x_lu - x_exact)
    print("LU error:", lu_error)

    # Метод Зейделя
    x0 = np.zeros(n)
    x_seidel, iterations, residual = seidel(A, F, x0)
    seidel_error = np.linalg.norm(x_seidel - x_exact)
    print("Seidel error:", seidel_error)

    return cond(A), gauss_error, lu_error, seidel_error


# Параметры для оценки
n_values = [3, 4, 5, 6]  # Размерности матрицы Гильберта

condition_numbers = []
gauss_errors = []
lu_errors = []
seidel_errors = []

# Оценка для каждой размерности n
for n in n_values:
    print("==== n =", n, "====")
    cond_number, gauss_error, lu_error, seidel_error = evaluate_methods(n)
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
plt.plot(n_values, gauss_errors, marker="o", label="Gauss Error")
plt.plot(n_values, lu_errors, marker="o", label="LU Error")
plt.plot(n_values, seidel_errors, marker="o", label="Seidel Error")
plt.title("Errors for Different Matrix Sizes")
plt.xlabel("Matrix Size (n)")
plt.ylabel("Error")
plt.legend()
plt.show()
