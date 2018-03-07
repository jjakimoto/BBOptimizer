from abc import ABCMeta, abstractmethod
from copy import deepcopy
import numpy as np

from ..space import DesignSpace
from .utils import register


class BaseSampler(object, metaclass=ABCMeta):
    def __init__(self, space, init_X=None, init_y=None):
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

    def update(self, new_X, new_y):
        self._X.extend(new_X)
        self._y.extend(new_y)
        self._update(new_X, new_y)

    @abstractmethod
    def _update(self, new_X, new_y):
        raise NotImplementedError()

    @property
    def data(self):
        return self._X, self._y

    def params2vec(self, params):
        # Not include fixed params
        vec = []
        for conf in self.params_conf:
            val = params[conf['name']]
            v = param2opt_space(val, conf['type'], conf["domain"])
            vec.append(v)
        return np.array(vec)

    def vec2params(self, vec):
        # Not include fixed params
        params = {}
        vec = self._design_space.model_to_objective(vec)
        for i, val in enumerate(vec):
            conf = self._design_space.config_space_expanded[i]
            if "scale" in conf and conf["scale"] == "log":
                val = 10 ** val
            if "is_integer" in conf and conf["is_integer"]:
                val = int(val)
        return params
