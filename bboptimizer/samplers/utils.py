from collections import defaultdict
import numpy as np
import random


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
