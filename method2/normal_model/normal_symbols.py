from typing import Sequence

from method1.normal_model.normal_symbols import method1_normal_fit
from symbols.normal import NormalSymbol


def method2_normal_fit(symbols: Sequence[NormalSymbol]) -> tuple[float, float]:
    # result is same as Method 1
    return method1_normal_fit(symbols)
