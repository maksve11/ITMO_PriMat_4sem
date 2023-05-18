# Наши функции, градиенты и их гессианы
def f1(x, y):
    return x ** 2 + y ** 2 - 4 * x


def f1_gradient(x, y):
    return [2 * x - 4, 2 * y]


def f1_hessian(x, y):
    return [[2, 0], [0, 2]]


def f2(x, y):
    return (6 * x - 3) ** 2 + (2 * y - 1) ** 2 - 3 * x * y


def f2_gradient(x, y):
    return [12 * x - 6 - 3 * y, 4 * y - 2 - 3 * x]


def f2_hessian(x, y):
    return [[12, -3], [-3, 4]]


def f3(x, y):
    return x ** 2 - y ** 2 + 2 * x * y + 2 * y - 3 * x


def f3_gradient(x, y):
    return [2 * x + 2 * y - 3, 2 * x - 2 * y + 2]


def f3_hessian(x, y):
    return [[2, 2], [2, -2]]