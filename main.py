import matplotlib.pyplot as plt

from cases.normal_normal import plot_normal_normal_method, plot_normal_normal_method_comparison
from cases.normal_uniform import plot_normal_uniform_method
from cases.uniform_normal import plot_uniform_normal_method, plot_uniform_normal_method_comparison
from cases.uniform_uniform import plot_uniform_uniform_fitting
from symbols.normal import NormalSymbol
from symbols.uniform import UniformSymbol


def main():
    method = 3

    # Uncomment whichever one is of interest
    uniform_normal(method=method)
    # uniform_uniform(method=method)
    # normal_uniform(method=method)
    # normal_normal(method=method)


def uniform_uniform(method: int | None = None):
    uniform_symbols = (
        UniformSymbol(-15, -5, 400),
        UniformSymbol(-6, 50, 43),
        UniformSymbol(10, 60, 2),
        UniformSymbol(30, 45, 12)
    )

    plot_uniform_uniform_fitting(uniform_symbols, method=method)
    plt.show()


def normal_uniform(method: int | None = None):
    uniform_symbols = (
        UniformSymbol(-15, 40, 4),
        UniformSymbol(-6, 50, 43),
        UniformSymbol(10, 60, 63),
        UniformSymbol(200, 210, 10)
    )

    plot_normal_uniform_method(uniform_symbols, method=method)

    plt.show()


def uniform_normal(method: int | None = None):
    normal_symbols = (
        NormalSymbol(-15, 6, 4),
        NormalSymbol(-6, 7, 43),
        NormalSymbol(10, 8, 2),
        NormalSymbol(30, 6.5, 12)
    )

    if method is None:
        plot_uniform_normal_method_comparison(normal_symbols, method1_precision=4)
    else:
        plot_uniform_normal_method(normal_symbols, method=method)

    plt.show()


def normal_normal(method: int | None = None):
    normal_symbols = (
        NormalSymbol(-15, 6, 4),
        NormalSymbol(-6, 7, 43),
        NormalSymbol(10, 8, 2),
        NormalSymbol(30, 6.5, 12)
    )

    if method is None:
        plot_normal_normal_method_comparison(normal_symbols)
    else:
        plot_normal_normal_method(normal_symbols, method=method)

    plt.show()


if __name__ == "__main__":
    main()
