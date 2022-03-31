import matplotlib.pyplot as plt

from cases.normal_normal import plot_normal_normal_fitting
from cases.normal_uniform import plot_normal_uniform_fitting
from cases.uniform_normal import plot_uniform_normal_fitting
from cases.uniform_uniform import plot_uniform_uniform_fitting
from symbols.normal import NormalSymbol
from symbols.uniform import UniformSymbol


def main():
    method = 3

    # Uncomment whichever one is of interest
    # uniform_normal(method=method)
    uniform_uniform(method=method)
    # normal_uniform(method=method)
    # normal_normal(method=method)


def uniform_uniform(method: int):
    # each column defines one symbol
    a = [-15, -6, 10, 30]
    b = [-5, 50, 60, 45]
    n = [400, 43, 2, 12]

    uniform_symbols = tuple(UniformSymbol(*params) for params in zip(a, b, n))

    plot_uniform_uniform_fitting(uniform_symbols, method=method)
    plt.show()


def normal_uniform(method: int):
    # each column makes one symbol
    a = [-15, -6, 10, 30, 200]
    b = [40, 50, 60, 45, 210]
    n = [4, 43, 63, 15, 10]

    uniform_symbols = tuple(UniformSymbol(*params) for params in zip(a, b, n))

    plot_normal_uniform_fitting(uniform_symbols, method=method)
    plt.show()


def uniform_normal(method: int):
    # each column makes one symbol
    m = [-15, -6, 10, 30]
    s = [6, 7, 8, 6.5]
    n = [4, 43, 2, 12]

    normal_symbols = tuple(NormalSymbol(*params) for params in zip(m, s, n))

    plot_uniform_normal_fitting(normal_symbols, method=method)
    plt.show()


def normal_normal(method: int):
    # each column makes one symbol
    m = [-15, -6, 10, 30]
    s = [6, 7, 8, 6.5]
    n = [4, 43, 2, 12]

    normal_symbols = tuple(NormalSymbol(*params) for params in zip(m, s, n))

    plot_normal_normal_fitting(normal_symbols, method=method)
    plt.show()


if __name__ == "__main__":
    main()
