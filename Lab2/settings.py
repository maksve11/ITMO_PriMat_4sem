import numpy as np
import seaborn as sns

sns.set_theme()

# Кастомные параметры оптимизации
A, B = 1, 20


# Наша функция
def f(x):
    if x <= 0:
        return float('inf')
    return np.sin(0.5 * np.log(x) * x) + 1


threshold = 1


def is_ratio_bellow_threshold(ratio) -> bool:
    return ratio > threshold


def is_work_correct(arr_ratio) -> bool:
    return len(arr_ratio) < 1 or is_ratio_bellow_threshold(arr_ratio[-1])

