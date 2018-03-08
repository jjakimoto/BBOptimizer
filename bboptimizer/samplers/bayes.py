# -*- coding: utf-8 -*-
# @Author: tom-hydrogen
# @Date:   2018-03-07 10:51:02
# @Last Modified by:   tom-hydrogen
# @Last Modified time: 2018-03-07 14:57:58
""" gp.py
Bayesian optimisation of loss functions.
"""
import numpy as np
from scipy.optimize import minimize
import GPy
from GPy.models import GPRegression, SparseGPRegression
from copy import deepcopy

from .core import BaseSampler
from .utils import random_sample, expected_improvement
from ..constants import EPSILON


class BayesSampler(BaseSampler):
    sampler_name = "bayes"

    def __init__(self, space, init_X=None, init_y=None, r_min=3, method="EI",
                 optimizer="bfgs", max_iters=1000,
                 is_normalize=True, ARD=True, kernel=None, sparse=False,
                 num_inducing=10):
        super(BayesSampler, self).__init__(space, init_X, init_y)
        self._r_min = r_min
        self.model = None
        self.acquisition_func = self._get_acquisition_func(method)
        self._is_normalize = is_normalize
        self._ARD = ARD
        self._kernel = kernel
        self._sparse = sparse
        self._num_inducing = num_inducing
        self._optimizer = optimizer
        self._max_iters = max_iters

    def _update(self, new_X, new_y, eps=EPSILON):
        X, y = self.data
        X = deepcopy(X)
        y = deepcopy(y)
        if len(X) >= self._r_min:
            X_vec = np.array([self.params2vec(x) for x in X])
            y = np.array(y)[:, None]
            if self._is_normalize:
                sig = np.sqrt(np.var(y))
                sig = max(sig, eps)
                mu = np.mean(y)
                y = (y - mu) / sig
            if self.model is None:
                self._create_model(X_vec, y)
            else:
                self.model.set_XY(X_vec, y)
            self.model.optimize(optimizer=self._optimizer,
                                max_iters=self._max_iters,
                                messages=False,
                                ipython_notebook=False)

    def _create_model(self, X, Y):
        """
        Creates the model given some input data X and Y.
        """

        # Define kernel
        self.input_dim = X.shape[1]
        if self._kernel is None:
            kern = GPy.kern.Matern52(self.input_dim, variance=1.,
                                     ARD=self._ARD)
        else:
            kern = self._kernel
            self._kernel = None

        # Define model
        noise_var = Y.var() * 0.01
        if not self._sparse:
            self.model = GPRegression(X, Y, kernel=kern,
                                      noise_var=noise_var)
        else:
            self.model = SparseGPRegression(X, Y, kernel=kern,
                                            num_inducing=self._num_inducing)
        self.model.Gaussian_noise.constrain_bounded(1e-9, 1e6, warning=False)

    def sample(self, num_samples=1, *args, **kwargs):
        _num_data = self.num_data
        if _num_data < self._r_min:
            Xs = self._random_sample(num_samples)
        else:
            Xs = self._bayes_sample(num_samples)
        return Xs

    def _bayes_sample(self, num_samples, num_restarts=25):
        num_restarts = max(num_samples, num_restarts)
        init_params = self._random_sample(num_restarts)
        init_xs = [self.params2vec(param) for param in init_params]
        bounds = self.design_space.get_bounds()
        evaluated_loss = np.array(self.model.Y)[:, 0]
        ys = []
        xs = []

        def minus_ac(x):
            return -self.acquisition_func(x, self.model,
                                          evaluated_loss)

        for x0 in init_xs:
            res = minimize(fun=minus_ac,
                           x0=x0,
                           bounds=bounds,
                           method='L-BFGS-B')
            ys.append(-res.fun)
            xs.append(res.x)
        idx = np.argsort(ys)[::-1][:num_samples]
        best_x = np.array(xs)[idx]
        best_params = [self.vec2params(x) for x in best_x]
        return best_params

    def _random_sample(self, num_samples):
        Xs = []
        for i in range(num_samples):
            x = random_sample(self.params_conf)
            Xs.append(x)
        return list(Xs)

    def _get_acquisition_func(self, method):
        if method == "EI":
            return expected_improvement
        else:
            raise NotImplementedError(method)

    def params2vec(self, params):
        # Not include fixed params
        vec = []
        for conf in self.params_conf:
            val = params[conf['name']]
            vec.append(val)
        vec = self.design_space.objective_to_model([vec])
        return np.array(vec)

    def vec2params(self, vec):
        # Not include fixed params
        params = {}
        vec = self.design_space.model_to_objective([vec])
        for i, val in enumerate(vec):
            conf = self.params_conf[i]
            params[conf["name"]] = val
        return params
