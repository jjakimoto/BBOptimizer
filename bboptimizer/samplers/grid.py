# -*- coding: utf-8 -*-
# @Author: tom-hydrogen
# @Date:   2018-03-07 15:18:34
# @Last Modified by:   tom-hydrogen
# @Last Modified time: 2018-03-07 15:57:06
from copy import deepcopy
from itertools import product
import numpy as np

from .core import BaseSampler


class GridSampler(BaseSampler):
    sampler_name = "grid"

    def __init__(self, space, init_X=None, init_y=None, num_grid=None):
        super(GridSampler, self).__init__(space, init_X, init_y)
        self.index = 0
        domains = []
        indices = []
        _params_conf = deepcopy(self.params_conf)
        # Set default grid
        for i, conf in enumerate(_params_conf):
            # Set default grid value
            if "num_grid" not in conf and num_grid is not None:
                if len(conf["domain"]) == 2:
                    conf["num_grid"] = num_grid
            # Configure domain
            domain = conf["domain"]
            if conf["type"] in ["continuous", "integer"]:
                if "num_grid" in conf:
                    scale = conf.get("scale", None)
                    if scale == 'log':
                        domain = np.logspace(domain[0],
                                             domain[1],
                                             conf["num_grid"])
                    else:
                        domain = np.linspace(domain[0],
                                             domain[1],
                                             conf["num_grid"])
                    if conf["type"] == "integer":
                        domain = domain.astype(int)
                else:
                    domain = tuple(domain)
            elif conf["type"] == "fixed":
                domain = (domain,)
            else:
                domain = tuple(domain)
            domains.append(list(domain))
            indices.append(i)

        # Sample parameters from parameters stored in self.params_list
        patterns = product(*domains)
        self.params_list = []
        for params_val in patterns:
            params_dict = dict()
            for i, idx in enumerate(indices):
                conf = _params_conf[idx]
                params_dict[conf["name"]] = params_val[i]
            self.params_list.append(params_dict)

    def sample(self, num_samples=1, *args, **kwargs):
        Xs = []
        for i in range(num_samples):
            x = self.params_list[self.index]
            Xs.append(x)
            self.index += 1
            self.index = self.index % len(self.params_list)
        return Xs
