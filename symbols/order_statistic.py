import numpy as np


class OrderStatisticSymbol:
    """
    Parameters:
    s_l - observed lower statistic value
    s_u - observed upper statistic value
    l - lower statistic ordinal, between 1 and u - 1 inclusive
    u - upper statistic ordinal, between l + 1 and n inclusive
    n - number of points summarised by symbol
    """

    def __init__(self, lower_stat: float, upper_stat: float, lower_order: int, upper_order: int, n: int):
        if not 1 <= lower_order < upper_order <= n:
            msg = f"Order statistic symbol must have 1 <= l < u <= nbut l={lower_order}, u={upper_order} and n={n}"
            raise ValueError(msg)
        elif not lower_stat <= upper_stat:
            raise ValueError(f"Order statistic symbol must have s_l <= s_u but s_l={lower_stat} and s_u={upper_stat}")
        elif not isinstance(n, int):
            raise TypeError(f"n must be integer but was {type(n)}")

        self.lower_stat = lower_stat
        self.upper_stat = upper_stat
        self.lower_order = lower_order
        self.upper_order = upper_order
        self.n = n

    # support tuple unpacking
    def __len__(self):
        return 5

    # support tuple unpacking
    def __getitem__(self, key):
        if type(key) == int:
            match key:
                case 0: return self.lower_order
                case 1: return self.upper_order
                case 2: return self.lower_stat
                case 3: return self.upper_stat
                case 4: return self.n
                case _:
                    raise IndexError("index out of range")
        else:
            raise TypeError(f"indices must be integers, not {type(key)}")

    def heuristic_min_max(self):
        # from maximising uniform-uniform symbolic likelihood in single-class case

        # a = (u * s_l - (l-1) * s_u) / (u - (l-1))
        #   = b - n * (b - s_u)/(n - u)
        # b = (s_u * (n - (l-1)) - s_l * (n-u) ) / (u - (l-1))
        #   = a + n * (s_l - a) / (l-1)

        l, u, n = self.lower_order, self.upper_order, self.n
        s_l, s_u = self.lower_stat, self.upper_stat,

        d = u - (l - 1)
        a = (u * s_l - (l - 1) * s_u) / d
        b = (s_u * (n - (l-1)) - s_l * (n-u)) / d
        return a, b

    def pdf(self, x: np.ndarray):
        raise TypeError("pdf not supported for order statistic symbol")

    def cdf(self, x: np.ndarray):
        raise TypeError("cdf not supported for order statistic symbol")

    def rvs(self):
        raise TypeError("rvs not supported for order statistic symbol")



