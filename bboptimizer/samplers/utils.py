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


def expected_improvement(x, model, evaluated_loss, eps=1e-5):
    """ expected_improvement
    Expected improvement acquisition function.

    Note
    ----
    This implementation aims for minimization

    Arguments:
    ----------
        x: array-like, shape = [n_hyperparams,]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
    """
    x = np.atleast_2d(x)
    mu, sigma2 = model.predict(x)
    # Consider 1d case
    sigma = np.sqrt(sigma2)[0, 0]
    mu = mu[0, 0]
    # Avoid too small sigma
    sigma = max(sigma, eps)
    loss_optimum = np.min(evaluated_loss)

    gamma = (loss_optimum - mu) / sigma
    ei_val = -sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
    return ei_val
