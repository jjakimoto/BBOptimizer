from .utils import random_sample
from .core import BaseSampler


class RandomSampler(BaseSampler):
    """Random optimization sampler

    Sample next location based on random sampling

    Parameters
    ----------
    space: list(dict)
        Define search space. Each element has to the following key
        values: 'name', 'type', and 'domain' (,'num_grid' is optional).
    init_X: array-like(float), shape=(n_samples, n_dim)
        The list of parameters to initizlie sampler
    init_y: array-like(float), shape(n_samples,)
        The list of score of init_X
    """

    sampler_name = "random"

    def sample(self, num_samples=1, *args, **kwargs):
        """Sample next location to evaluate at random

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
            x = random_sample(self.params_conf)
            Xs.append(x)
        return Xs

    def _update(self, new_X, new_y):
        # Random sampling does not depend on historical results
        pass
