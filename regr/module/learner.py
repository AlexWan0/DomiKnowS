import abc
from typing import Any
from .module import Module

class Learner(Module):
    __metaclass__ = abc.ABCMeta

    @property
    @abc.abstractmethod
    def parameters(self) -> Any: pass
