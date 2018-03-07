from .utils import random_sample
from .core import BaseSampler


class RandomSampler(BaseSampler):
    sampler_name = "random"

    def sample(self, num_samples=1, *args, **kwargs):
        Xs = []
        for i in range(num_samples):
            x = random_sample(self.params_conf)
            Xs.append(x)
        return Xs

    def _update(self, new_X, new_y):
        # Random sampling does not depend on historical results
        pass
