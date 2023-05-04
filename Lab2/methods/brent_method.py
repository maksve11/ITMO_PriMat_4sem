import numpy as np
import settings


def brent(eps, f, a, b):
    arr_a, arr_b, arr_ratio = [a], [b], []
    count_iter, count_func = 0, 0

    eps1 = eps / 10
    phi = (3 - np.sqrt(5)) / 2
    x = w = v = a + phi * (b - a)
    fx = fw = fv = f(x)
    d, e = b - a, b - a

    while d > eps:
        g = e
        e = d

        u = None
        if x != w and x != v and w != v and fx != fw and fx != fv and fw != fv:
            p = (x - w) * (fx - fv) - (x - v) * (fx - fw)
            q = 2 * (x - w) * (fx - fv) - 2 * (x - v) * (fx - fw)

            if q != 0:
                u = x - (p / q)
                if a + eps1 <= u <= b - eps1 and np.abs(u - x) < g / 2:
                    d = np.abs(u - x)
                else:
                    u = None

        if u is None:
            if x < (b + a) / 2:
                u = x + phi * (b - x)
                d = b - x
            else:
                u = x - phi * (x - a)
                d = x - a

            if np.abs(u - x) < eps1:
                u = x + np.sign(u - x) * eps1

        fu = f(u)
        count_func += 1

        if fu <= fx:
            if u >= x:
                a = x
            else:
                b = x

            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u >= x:
                b = u
            else:
                a = u

            if fu <= fw or w == x:
                v, w = w, u
                fv, fw = fw, fu
            elif fu <= fv or v == x:
                v, fv = u, fu

        arr_a.append(a), arr_b.append(b)
        arr_ratio.append(e / d)
        count_iter += 1

    return x, count_iter, count_func, arr_a, arr_b, arr_ratio


print("\n===============Brent===============")
# Tests
val = brent(0.001, settings.f, settings.A, settings.B)[0]
print(f"Алгоритм нашел минимум функции в точке {val}")

val = brent(0.005, np.sin, -2.5, 2)[0]
print(f"Алгоритм нашел минимум функции sin(x) в точке {val}")

val = brent(0.005, np.cos, -2.5, 2)[0]
print(f"Алгоритм нашел минимум функции cos(x) в точке {val}")
