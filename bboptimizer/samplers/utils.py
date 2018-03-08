from collections import defaultdict
import numpy as np
import random
from scipy.stats import norm
from scipy.special import erfc


SAMPLERS_MAP = dict()


def register(cls):
    SAMPLERS_MAP[cls.sampler_name] = cls


def _random_sample(params_conf):
    """Sample parameters at random with dictionary format

    Parameters
    ----------
    params_conf: list(dict) or dict

    Returns
    -------
    params_dict : dict
        key is a name of each parameter
    """
    if len(params_conf) == 0 or not isinstance(params_conf[0], list):
        params_conf = [params_conf]
    params_conf = random.sample(params_conf, k=1)[0]
    params_dict = {}
    for conf in params_conf:
        name = conf["name"]
        domain = conf["domain"]
        type_ = conf["type"]
        scale = conf.get('scale', None)
        if type_ == "continuous":
            if scale == 'log':
                param = np.random.uniform(np.log10(domain[0]),
                                          np.log10(domain[1]))
                param = 10 ** param
            else:
                param = np.random.uniform(domain[0], domain[1])
        elif type_ == "integer":
            if scale == 'log':
                param = np.random.uniform(np.log10(domain[0]),
                                          np.log10(domain[1]))
                param = round(10 ** param)
            else:
                param = np.random.randint(domain[0], domain[1] + 1)
        elif type_ in ["categorical", "discrete"]:
            param = random.sample(domain, k=1)[0]
        elif type_ == "fixed":
            param = domain
        params_dict[name] = param
    return params_dict


def random_sample(params_conf, x=None):
    if x is None:
        x = defaultdict(lambda: None)
    if isinstance(params_conf, dict):
        for key, conf in params_conf.items():
            x[key] = random_sample(conf, x[key])
    else:
        x = _random_sample(params_conf)
    return x

"""
def expected_improvement(x, model, evaluated_loss, jitter=0.01):
    expected_improvement
    Expected improvement acquisition function.

    Note
    ----
    This implementation aims for minimization

    Parameters:
    ----------
    x: array-like, shape = (n_hyperparams,)
    model: GaussianProcessRegressor object.
        Gaussian process trained on previously evaluated hyperparameters.
    evaluated_loss: array-like(float), shape = (# historical results,).
         the values of the loss function for the previously evaluated
         hyperparameters.
    jitter: float
        positive value to make the acquisition more explorative.

    x = np.atleast_2d(x)
    mu, var = model.predict(x)
    # Consider 1d case
    sigma = np.sqrt(var)[0, 0]
    mu = mu[0, 0]
    # Avoid too small sigma
    if sigma == 0.:
        return 0.
    else:
        loss_optimum = np.min(evaluated_loss)
        gamma = (loss_optimum - mu) / sigma
        ei_val = sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
        return ei_val
"""


def get_quantiles(acquisition_par, fmin, m, s):
    '''
    Quantiles of the Gaussian distribution useful to determine the acquisition function values
    :param acquisition_par: parameter of the acquisition function
    :param fmin: current minimum.
    :param m: vector of means.
    :param s: vector of standard deviations.
    '''
    if isinstance(s, np.ndarray):
        s[s<1e-10] = 1e-10
    elif s< 1e-10:
        s = 1e-10
    u = (fmin-m-acquisition_par)/s
    phi = np.exp(-0.5 * u**2) / np.sqrt(2*np.pi)
    Phi = 0.5 * erfc(-u / np.sqrt(2))
    return (phi, Phi, u)


def expected_improvement(x, model, evaluated_loss, jitter=0.01):
    """ expected_improvement
    Expected improvement acquisition function.

    Note
    ----
    This implementation aims for minimization

    Parameters:
    ----------
    x: array-like, shape = (n_hyperparams,)
    model: GaussianProcessRegressor object.
        Gaussian process trained on previously evaluated hyperparameters.
    evaluated_loss: array-like(float), shape = (# historical results,).
         the values of the loss function for the previously evaluated
         hyperparameters.
    jitter: float
        positive value to make the acquisition more explorative.
    """
    x = np.atleast_2d(x)
    mu, var = model.predict(x)
    # Consider 1d case
    sigma = np.sqrt(var)[0, 0]
    mu = mu[0, 0]
    # Avoid too small sigma
    if sigma == 0.:
        return 0.
    else:
        loss_optimum = np.min(evaluated_loss)
        phi, Phi, u = get_quantiles(jitter, loss_optimum, mu, sigma)
        f_acqu = sigma * (u * Phi + phi)
        return f_acqu

    def _compute_acq(self, x):
        """
        Computes the Expected Improvement per unit of cost
        """
        m, s = self.model.predict(x)
        fmin = self.model.get_fmin()
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu = s * (u * Phi + phi)
        return f_acqu
