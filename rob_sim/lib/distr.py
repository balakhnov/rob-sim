import numpy as np
from abc import ABC, abstractmethod


class Distribution(ABC):
    @abstractmethod
    def sample(self):
        pass


class UniformDistr(Distribution):
    def __init__(self, low, high, shape = None) -> None:
        self.low = low
        self.high = high
        self.shape = shape

    def sample(self):
        return np.random.uniform(self.low, self.high, self.shape)
         

class NormDistr(Distribution):
    def __init__(self, loc, scale, shape) -> None:
        self.loc = loc
        self.scale = scale
        self.shape = shape

    def sample(self):
        return np.random.normal(self.loc, self.scale, self.shape)


class FixedVal(UniformDistr):
    def __init__(self, val, shape=None) -> None:
        super().__init__(val, val, shape)

    def sample(self):
        return super().sample()
