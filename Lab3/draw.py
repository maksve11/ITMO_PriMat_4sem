import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def draw_func(a, b, func, func_gradient, points_x, points_y, title):
    fig, ax = plt.subplots()
    x, y = np.mgrid[a:b:100j, a:b:100j]
    z = func(x, y)
    ax.contour(x, y, z, levels=100, colors='green')
    ax.set_title(title)
    for i in range(len(points_x)):
        ax.scatter(points_x[i], points_y[i], c='black')
    ax.plot(points_x, points_y, c='black')

    # Градиент
    gradient_x, gradient_y = func_gradient(points_x[-1], points_y[-1])
    ax.quiver(points_x[-1], points_y[-1], gradient_x, gradient_y, scale=20, color='red')

    plt.show()


def draw_func_hessian(a, b, hessian_function):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    x, y = np.mgrid[a:b:100j, a:b:100j]
    z = np.zeros_like(x)  # Создаем массив нулей той же размерности, что и x, y

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            hessian = hessian_function(x[i, j], y[i, j])
            eigvals = np.linalg.eigvals(hessian)
            z[i, j] = min(eigvals)

    ax.contour(x, y, z, levels=100, colors='firebrick')
    plt.show()
