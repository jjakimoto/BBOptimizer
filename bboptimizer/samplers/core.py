from abc import ABCMeta, abstractmethod
from copy import deepcopy

from ..space import DesignSpace
from .utils import register


class BaseSampler(object, metaclass=ABCMeta):
    def __init__(self, space, init_X=None, init_y=None, *args, **kwargs):
        self._space = deepcopy(space)
        self.design_space = DesignSpace(space)
        if init_X is None:
            init_X = []
        self._X = init_X
        if init_y is None:
            init_y = []
        self._y = init_y

    @classmethod
    def __init_subclass__(cls):
        if hasattr(cls, "sampler_name"):
            register(cls)
        return cls

    @abstractmethod
    def sample(self, *args, **kwargs):
        raise NotImplementedError()

    def update(self, new_X=None, new_y=None):
        if new_X is None:
            new_X = []
        if new_y is None:
            new_y = []
        self._X.extend(new_X)
        self._y.extend(new_y)
        self._update(new_X, new_y)

    @abstractmethod
    def _update(self, new_X, new_y):
        raise NotImplementedError()

    @property
    def data(self):
        return self._X, self._y

    @property
    def num_data(self):
        return len(self._X)

    @property
    def params_conf(self):
        return self.design_space.config_space_expanded
