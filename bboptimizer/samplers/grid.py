# -*- coding: utf-8 -*-
# @Author: tom-hydrogen
# @Date:   2018-03-07 15:18:34
# @Last Modified by:   tom-hydrogen
# @Last Modified time: 2018-03-09 16:52:07
from copy import deepcopy
from itertools import product
import numpy as np

from .core import BaseSampler


class GridSampler(BaseSampler):
    """Grid optimization sampler

    Sample next location based on grid sampling

    Parameters
    ----------
    space: list(dict)
        Define search space. Each element has to the following key
        values: 'name', 'type', and 'domain' (,'num_grid' is optional).
    init_X: array-like(float), shape=(n_samples, n_dim)
        The list of parameters to initizlie sampler
    init_y: array-like(float), shape(n_samples,)
        The list of score of init_X
    num_grid: int, optional
        The default number of grid
    """

    sampler_name = "grid"

    def __init__(self, space, init_X=None, init_y=None, num_grid=None,
                 *args, **kwargs):
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
        """Sample next location to evaluate based on grid.

        Everytime this function is called, it samples points not sampled yet.

        Parameters
        ---------
        num_samples: int
            The number of samples

        Returns
        -------
        Xs: list(dict), length is num_samples
        """
        Xs = []
        for i in range(num_samples):
            x = self.params_list[self.index]
            Xs.append(x)
            self.index += 1
            self.index = self.index % len(self.params_list)
        return Xs
