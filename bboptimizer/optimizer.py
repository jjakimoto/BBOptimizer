from copy import deepcopy
import numpy as np
from multiprocessing import Process, Queue

from .exceptions import TimelimitError
from .samplers.utils import SAMPLERS_MAP


class Optimizer(object):
    """Abstract class of black box optimization

    Black box optimization is a method to find optimal parameteres by
    observing the response of an objective function wrt each parameters
    without defining models.
    Cases where target values are not given are considered as application
    , e.g., hyperparameter optimization.

    Parameters
    ----------
    score_func : function
        Takes dictionary as input and returns scalar score.
    params_conf : list(dict)
        Each element define name search space as a dictionary.
    init_params : list(dict), optional
        List of parameters to initialize.
    init_scores : list(float), optional
        List of scores for init_params. Note that the length has to be the same with that of init_params.
    """

    def __init__(self, score_func, space,
                 sampler="random", init_X=None, init_y=None,
                 timeout=None, *args, **kwargs):
        self._score_func = score_func
        self._space_conf = space
        self._timeout = timeout
        # Separate fixed params
        self.fixed_params = dict()
        self._nonfixed_conf = []
        if isinstance(self._space_conf, list):
            for conf in self._space_conf:
                if conf["type"] == "fixed":
                    self.fixed_params[conf["name"]] = conf["domain"]
                else:
                    self._nonfixed_conf.append(conf)
        else:
            self._nonfixed_conf = self._space_conf
        # Sampler cares only about non fixed parameters
        if isinstance(init_X, list):
            for param in init_X:
                for fixed_name in self.fixed_params.keys():
                    del param[fixed_name]
        self._sampler = SAMPLERS_MAP[sampler](space, init_X, init_y)

    def search(self, return_full=False, num_iter=10, *args, **kwargs):
        for i in range(num_iter):
            Xs = self._sampler.sample(*args, **kwargs)
            sucess_Xs = []
            ys = []
            for X in Xs:
                try:
                    y = self.score_func(X)
                    ys.append(y)
                    sucess_Xs.append(X)
                except TimelimitError as e:
                    print(e)
                    print("Try different configuration")
                    continue
            self._sampler.update(sucess_Xs, ys)
        Xs, ys = self._sampler.data
        # Update with  fixed parameters
        fixed_params = deepcopy(self.fixed_params)
        for X in Xs:
            X.update(fixed_params)
        if return_full:
            return Xs, ys
        else:
            best_idx = np.argmin(ys)
            best_X = Xs[best_idx]
            best_y = ys[best_idx]
            return best_X, best_y

    def score_func(self, X, *args, **kwargs):
        fixed_params = deepcopy(self.fixed_params)
        X.update(fixed_params)

        def record(que):
            try:
                score = self._score_func(X, *args, **kwargs)
            except Exception as e:
                score = e
            que.put(score)

        que = Queue()
        proc = Process(target=record, args=(que,))
        proc.start()
        if self._timeout is not None:
            proc.join(self.timeout)
        else:
            proc.join()
        if proc.is_alive():
            proc.terminate()
            proc.join()
            raise TimelimitError()
        else:
            response = que.get()
            if isinstance(response, Exception):
                print("Error at score_func", response)
                raise response
            return response
