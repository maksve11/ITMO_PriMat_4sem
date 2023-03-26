def get_leftsum(f, arguments, h):
    return sum(h * f(arg) for arg in arguments[:-1])


def get_rightsum(f, arguments, h):
    return sum(h * f(arg) for arg in arguments[1:])


def get_centersum(f, arguments, h):
    return sum(h * f((arguments[i] + arguments[i + 1]) / 2) for i in range(len(arguments) - 1))


def get_trapezoidsum(f, arguments, h):
    argf = [f(arg) for arg in arguments]
    return sum(h * (argf[i] + argf[i + 1]) / 2 for i in range(len(argf) - 1))


def get_parabolicsum(f, arguments, h):
    return sum(h * (f(arguments[i]) + f(arguments[i + 1]) + 4 * f((arguments[i] + arguments[i + 1]) / 2)) / 6 for i in range(len(arguments) - 1))
