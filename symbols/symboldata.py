from abc import ABC
from typing import Iterable


class SymbolData(ABC):
    # class used to represent data in a symbol
    def __init__(self, values: Iterable[float]):
        self.values = tuple(v for v in values)

    @property
    def num(self):
        return len(self)

    # support tuple unpacking
    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, item: int):
        return self.values[item]
