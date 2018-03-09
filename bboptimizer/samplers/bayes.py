# -*- coding: utf-8 -*-
# @Author: tom-hydrogen
# @Date:   2018-03-07 10:51:02
# @Last Modified by:   tom-hydrogen
# @Last Modified time: 2018-03-09 16:51:22
""" gp.py
Bayesian optimisation of loss functions.
"""
import numpy as np
from scipy.optimize import minimize
from copy import deepcopy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import GPy
from GPy.models import GPRegression, SparseGPRegression

from .core import BaseSampler
from .utils import random_sample, expected_improvement
from ..constants import EPSILON, RANDOM_STATE


class BayesSampler(BaseSampler):
    """Bayesian optimization sampler

    Sample next location based on gaussian process

    Parameters
    ----------
    space: list(dict)
        Define search space. Each element has to the following key
        values: 'name', 'type', and 'domain' (,'num_grid' is optional).
    init_X: array-like(float), shape=(n_samples, n_dim)
        The list of parameters to initizlie sampler
    init_y: array-like(float), shape(n_samples,)
        The list of score of init_X
    r_min: int
        The number of random samples before starting using gaussian process
    method: str
        The name of acquisition functions
    kernel: kernel object, optional
    is_normalize: bool
        If ture, normalized score values are used for optimization
    n_restarts_optimizer: int
        The number of trial to opimize GP hyperparameters
    backend: str (default sklearn)
        Determine which GP package is used. That has to be
        either of 'gpy' or 'sklearn'.
    optimizer: str
        The name of optimizers of hyperparameters of GP, which is valid when
        backend='gpy'.
    max_iters: int
        The maximum number of iteration to optimize hyperparamters of GP,
        which is valid when backend='gpy'.
    ARD: bool
        Wheather to use ARD for kernel, which is valid when backend='gpy'.
    sparse: bool
        If true, use sparse GP, which is valid when backend='gpy'.
    num_inducing: int
        The number of inducing inputs for sparse GP, which is valid
        when backend='gpy' and sparse is True.
    random_state: int
    """

    sampler_name = "bayes"

    def __init__(self, space, init_X=None, init_y=None, r_min=3, method="EI",
                 kernel=None, is_normalize=True, n_restarts_optimizer=10,
                 backend="sklearn", optimizer="bfgs", max_iters=1000,
                 ARD=False, sparse=False,
                 num_inducing=10, random_state=RANDOM_STATE):
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
        self._backend = backend.lower()
        self._n_restarts_optimizer = n_restarts_optimizer
        self._random_state = random_state

    def _update(self, new_X, new_y, eps=EPSILON):
        X, y = self.data
        X = deepcopy(X)
        y = deepcopy(y)
        if len(X) >= self._r_min:
            X_vec = np.array([self.params2vec(x) for x in X])
            y = np.array(y)
            if self._is_normalize:
                sig = np.sqrt(np.var(y))
                sig = max(sig, eps)
                mu = np.mean(y)
                y = (y - mu) / sig
            if self._backend == "sklearn":
                if self._kernel is None:
                    self._kernel = Matern(nu=2.5)
                self.model = GaussianProcessRegressor(
                    kernel=self._kernel,
                    n_restarts_optimizer=self._n_restarts_optimizer,
                    random_state=self._random_state,
                    normalize_y=False
                )
                self.model.fit(X_vec, y)
            elif self._backend.lower() == "gpy":
                y = np.array(y)[:, None]
                if self.model is None:
                    self._create_model(X_vec, y)
                else:
                    self.model.set_XY(X_vec, y)
                self.model.optimize_restarts(self._n_restarts_optimizer,
                                             optimizer=self._optimizer,
                                             max_iters=self._max_iters,
                                             messages=False,
                                             verbose=False,
                                             ipython_notebook=False)

    def _create_model(self, X, y):
        """
        Creates the GPy model given some input data X and Y.
        """

        # Define kernel
        input_dim = X.shape[1]
        if self._kernel is None:
            kern = GPy.kern.Matern52(input_dim, variance=1., ARD=self._ARD)
        else:
            kern = self._kernel
            self._kernel = None

        # Define model
        noise_var = y.var() * 0.01
        if not self._sparse:
            self.model = GPRegression(X, y, kernel=kern, noise_var=noise_var)
        else:
            self.model = SparseGPRegression(X, y, kernel=kern,
                                            num_inducing=self._num_inducing)
        self.model.Gaussian_noise.constrain_bounded(1e-9, 1e6, warning=False)

    def sample(self, num_samples=1, *args, **kwargs):
        """Sample next location to evaluate based on GP

        Parameters
        ---------
        num_samples: int
            The number of samples

        Returns
        -------
        Xs: list(dict), length is num_samples
        """
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
        if self._backend == "sklearn":
            evaluated_loss = np.array(self.model.y_train_)
        else:
            evaluated_loss = np.array(self.model.Y)[:, 0]

        ys = []
        xs = []

        def minus_ac(x):
            return -self.acquisition_func(x, self.model,
                                          evaluated_loss,
                                          mode=self._backend)

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
