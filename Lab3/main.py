from draw import draw_func, draw_func_hessian
from gradient_descent import conjugate_gradient_method
from quadratic_funcs import f1, f1_gradient, f1_hessian, f2_hessian

initial_point = [0, 0]
epsilon = 1e-6
max_iterations = 1000

optimal_point, _ = conjugate_gradient_method(f1, f1_gradient, f1_hessian, initial_point, epsilon, max_iterations)

points_x = [initial_point[0], optimal_point[0]]
points_y = [initial_point[1], optimal_point[1]]

draw_func(-50, 50, f1, f1_gradient, points_x, points_y, "f1 Optimization")
# draw_func_hessian(-50, 50, f1_hessian)
