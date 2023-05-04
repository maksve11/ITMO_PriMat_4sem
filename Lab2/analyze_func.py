import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns
from methods.brent_method import brent
from methods.dichotomy_method import dichotomy
from methods.fibonacci_method import fibonacci
from methods.golden_ratio_method import golden_ratio
from settings import f, A, B
from methods.parabolas_method import parabolas


def plot_graph(ax, x, y, xlabel, ylabel, title, label):
    sns.lineplot(x=x, y=y, ax=ax, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_scatter(ax, x, y, xlabel, ylabel, title, label):
    sns.scatterplot(x=x, y=y, ax=ax, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def analyze(f, a, b):
    stats = {"golden_ratio": {
        "count_iter": [],
        "count_func": [],
        "arr_a": [],
        "arr_b": [],
        "arr_ratio": [],
        "eps": []
    }, "dichotomy": {
        "count_iter": [],
        "count_func": [],
        "arr_a": [],
        "arr_b": [],
        "arr_ratio": [],
        "eps": []
    }, "fibonacci": {
        "count_iter": [],
        "count_func": [],
        "arr_a": [],
        "arr_b": [],
        "arr_ratio": [],
        "eps": []
    }, "parabolas": {
        "count_iter": [],
        "count_func": [],
        "arr_a": [],
        "arr_b": [],
        "arr_ratio": [],
        "eps": []
    }, "brent": {
        "count_iter": [],
        "count_func": [],
        "arr_a": [],
        "arr_b": [],
        "arr_ratio": [],
        "eps": []
    }}

    for eps in tqdm(np.linspace(0.0001, 0.1, 101)):
        # Dichotomy
        output = dichotomy(eps, f, a, b)
        stats["dichotomy"]["count_iter"].append(output[1])
        stats["dichotomy"]["count_func"].append(output[2])
        stats["dichotomy"]["arr_a"].append(output[3])
        stats["dichotomy"]["arr_b"].append(output[4])
        stats["dichotomy"]["arr_ratio"].append(output[5])
        stats["dichotomy"]["eps"].append(eps)

        # Golden ratio
        output = golden_ratio(eps, f, a, b)
        stats["golden_ratio"]["count_iter"].append(output[1])
        stats["golden_ratio"]["count_func"].append(output[2])
        stats["golden_ratio"]["arr_a"].append(output[3])
        stats["golden_ratio"]["arr_b"].append(output[4])
        stats["golden_ratio"]["arr_ratio"].append(output[5])
        stats["golden_ratio"]["eps"].append(eps)

        # Fibonacci
        output = fibonacci(eps, f, a, b)
        stats["fibonacci"]["count_iter"].append(output[1])
        stats["fibonacci"]["count_func"].append(output[2])
        stats["fibonacci"]["arr_a"].append(output[3])
        stats["fibonacci"]["arr_b"].append(output[4])
        stats["fibonacci"]["arr_ratio"].append(output[5])
        stats["fibonacci"]["eps"].append(eps)

        # Parabolas
        output = parabolas(eps, f, a, b)
        stats["parabolas"]["count_iter"].append(output[1])
        stats["parabolas"]["count_func"].append(output[2])
        stats["parabolas"]["arr_a"].append(output[3])
        stats["parabolas"]["arr_b"].append(output[4])
        stats["parabolas"]["arr_ratio"].append(output[5])
        stats["parabolas"]["eps"].append(eps)

        # Brent's method
        output = brent(eps, f, a, b)
        stats["brent"]["count_iter"].append(output[1])
        stats["brent"]["count_func"].append(output[2])
        stats["brent"]["arr_a"].append(output[3])
        stats["brent"]["arr_b"].append(output[4])
        stats["brent"]["arr_ratio"].append(output[5])
        stats["brent"]["eps"].append(eps)

    fig, ax = plt.subplots(3, 2, figsize=(20, 15))

    sns.lineplot(x=stats["golden_ratio"]["eps"], y=stats["golden_ratio"]["count_iter"], ax=ax[0][0],
                 label="Golden ratio")
    sns.lineplot(x=stats["dichotomy"]["eps"], y=stats["dichotomy"]["count_iter"], ax=ax[0][0], label="Dichotomy")
    sns.lineplot(x=stats["fibonacci"]["eps"], y=stats["fibonacci"]["count_iter"], ax=ax[0][0], label="Fibonacci")
    sns.lineplot(x=stats["parabolas"]["eps"], y=stats["parabolas"]["count_iter"], ax=ax[0][0], label="Parabolas")
    sns.lineplot(x=stats["brent"]["eps"], y=stats["brent"]["count_iter"], ax=ax[0][0], label="Brent")
    ax[0][0].set_xlabel("Точность")
    ax[0][0].set_ylabel("Количество итераций")
    ax[0][0].set_title("Зависимость количества итераций от точности")

    sns.lineplot(x=stats["golden_ratio"]["eps"], y=stats["golden_ratio"]["count_func"], ax=ax[0][1],
                 label="Golden ratio")
    sns.lineplot(x=stats["dichotomy"]["eps"], y=stats["dichotomy"]["count_func"], ax=ax[0][1], label="Dichotomy")
    sns.lineplot(x=stats["fibonacci"]["eps"], y=stats["fibonacci"]["count_func"], ax=ax[0][1], label="Fibonacci")
    sns.lineplot(x=stats["parabolas"]["eps"], y=stats["parabolas"]["count_func"], ax=ax[0][1], label="Parabolas")
    sns.lineplot(x=stats["brent"]["eps"], y=stats["brent"]["count_func"], ax=ax[0][1], label="Brent")
    ax[0][1].set_xlabel("Точность")
    ax[0][1].set_ylabel("Количество вызовов функции")
    ax[0][1].set_title("Зависимость количества вызовов функции от точности")

    sns.scatterplot(x=np.arange(1, stats["golden_ratio"]["count_iter"][0] + 2), y=stats["golden_ratio"]["arr_a"][0],
                    ax=ax[1][0], label="Golden ratio")
    sns.scatterplot(x=np.arange(1, stats["dichotomy"]["count_iter"][0] + 1), y=stats["dichotomy"]["arr_a"][0],
                    ax=ax[1][0], label="Dichotomy")
    sns.scatterplot(x=np.arange(1, stats["fibonacci"]["count_iter"][0] + 1), y=stats["fibonacci"]["arr_a"][0],
                    ax=ax[1][0], label="Fibonachi")
    sns.scatterplot(x=np.arange(1, stats["parabolas"]["count_iter"][0] + 2), y=stats["parabolas"]["arr_a"][0],
                    ax=ax[1][0], label="Parabolas")
    sns.scatterplot(x=np.arange(1, stats["brent"]["count_iter"][0] + 2), y=stats["brent"]["arr_a"][0], ax=ax[1][0],
                    label="Brent")
    ax[1][0].set_xlabel("Количество итераций")
    ax[1][0].set_ylabel("Левая граница")
    ax[1][0].set_title("Зависимость левой границы от итерации для eps = 0.0001")

    sns.scatterplot(x=np.arange(1, stats["golden_ratio"]["count_iter"][0] + 2), y=stats["golden_ratio"]["arr_b"][0],
                    ax=ax[1][1], label="Golden ratio")
    sns.scatterplot(x=np.arange(1, stats["dichotomy"]["count_iter"][0] + 1), y=stats["dichotomy"]["arr_b"][0],
                    ax=ax[1][1], label="Dichotomy")
    sns.scatterplot(x=np.arange(1, stats["fibonacci"]["count_iter"][0] + 1), y=stats["fibonacci"]["arr_b"][0],
                    ax=ax[1][1], label="Fibonacci")
    sns.scatterplot(x=np.arange(1, stats["parabolas"]["count_iter"][0] + 2), y=stats["parabolas"]["arr_b"][0],
                    ax=ax[1][1], label="Parabolas")
    sns.scatterplot(x=np.arange(1, stats["brent"]["count_iter"][0] + 2), y=stats["brent"]["arr_b"][0], ax=ax[1][1],
                    label="Brent")
    ax[1][1].set_xlabel("Количество итераций")
    ax[1][1].set_ylabel("Правая граница")
    ax[1][1].set_title("Зависимость правой границы от итерации для eps = 0.0001")

