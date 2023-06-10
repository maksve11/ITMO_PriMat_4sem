import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import cond

from main import generate_matrix, generate_rhs, exact_solution, lu_decomposition, solve_lu
from methods.seidel import seidel


def evaluate_methods(n, k_values):
    gauss_errors = []
    lu_errors = []
    seidel_errors = []

    for k in k_values:
        A = generate_matrix(n, k)
        F = generate_rhs(n)
        x_exact = exact_solution(n)

        print("==== k =", k, "====")
        print("Matrix A^(k):")
        print(A)
        print("Condition number:", cond(A))

        # Метод Гаусса с выбором ведущего элемента
        x_gauss = np.linalg.solve(A, F)
        gauss_error = np.linalg.norm(x_gauss - x_exact)
        gauss_errors.append(gauss_error)
        print("Gauss solution:", x_gauss)
        print("Gauss error:", gauss_error)

        # Метод LU-разложения
        L, U = lu_decomposition(A)
        x_lu = solve_lu(L, U, F)
        lu_error = np.linalg.norm(x_lu - x_exact)
        lu_errors.append(lu_error)
        print("LU solution:", x_lu)
        print("LU error:", lu_error)

        # Метод Зейделя
        x0 = np.zeros(n)
        x_seidel, iterations, residual = seidel(A, F, x0)
        seidel_error = np.linalg.norm(x_seidel - x_exact)
        seidel_errors.append(seidel_error)
        print("Seidel solution:", x_seidel)
        print("Seidel error:", seidel_error)
        print("Number of iterations:", iterations)
        print("Residual:", residual)
        print()

    # График ошибок для метода Гаусса
    plt.plot(k_values, gauss_errors, marker="o")
    plt.title("Gauss Error")
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.show()

    # График ошибок для метода LU-разложения
    plt.plot(k_values, lu_errors, marker="o")
    plt.title("LU Error")
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.show()

    # График ошибок для метода Зейделя
    plt.plot(k_values, seidel_errors, marker="o")
    plt.title("Seidel Error")
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.show()


# Параметры для исследования
n = 5  # Размерность матрицы
k_values = [1, 2, 3]  # Значения k для изменения обусловленности

# Исследование методов для различных значений k
evaluate_methods(n, k_values)
