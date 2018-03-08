from collections import defaultdict
import numpy as np
import random
from scipy.stats import norm


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


def expected_improvement(x, model, evaluated_loss, jitter=0.01, mode="gpy"):
    """Expected improvement acquisition function.

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
    if mode == "gpy":
        mu, var = model.predict(x)
        # Consider 1d case
        sigma = np.sqrt(var)[0, 0]
        mu = mu[0, 0]
    else:
        mu, sig = model.predict(x, return_std=True)
        mu = mu[0]
        sigma = sig[0]
    # Avoid too small sigma
    if sigma == 0.:
        return 0.
    else:
        loss_optimum = np.min(evaluated_loss)
        gamma = (loss_optimum - mu - jitter) / sigma
        ei_val = sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
        return ei_val
