import unittest
import numpy as np
import random
from copy import deepcopy

from bboptimizer import Optimizer
from bboptimizer.constants import RANDOM_STATE


map_func = dict(linear=lambda x: x, square=lambda x: x**2, sin=np.sin)


def test_func(x):
    print("x", x)
    score = np.sin(x["x2_1"]) + map_func[x["x3"]](x["x1"]) + map_func[x["x3"]](x["x2_2"])
    score_val = -score
    print("score_val", score_val)
    return score_val


params_conf = [
    {"name": "x1", "domain": (.1, 5), "type": "continuous",
     "num_grid": 5, "scale": "log"},
    {"name": "x2", "domain": (-5, 5), "type": "continuous",
     "num_grid": 5, "dimensionality": 2},
    {"name": "x3", "domain": ("linear", "sin", "square"),
     "type": "categorical"},
]


class TestBayesOptimizer(unittest.TestCase):
    init_params = dict(r_min=5)
    search_params = dict(num_iter=30)

    def test_search(self):
        self.init_params['score_func'] = test_func
        self.init_params['space'] = deepcopy(params_conf)
        bayes_optimizer = Optimizer(sampler="bayes", backend="gpy",
                                    **self.init_params)
        # Search parameters
        np.random.seed(RANDOM_STATE)
        random.seed(RANDOM_STATE)
        bayes_params, bayes_score =\
            bayes_optimizer.search(**self.search_params)

        grid_optimizer = Optimizer(sampler="random", **self.init_params)
        # Search parameters
        np.random.seed(RANDOM_STATE)
        random.seed(RANDOM_STATE)

        grid_params, grid_score =\
            grid_optimizer.search(**self.search_params)
        print(bayes_score, grid_score)
        self.assertTrue(bayes_score - grid_score <= 0)


if __name__ == '__main__':
    unittest.main()
