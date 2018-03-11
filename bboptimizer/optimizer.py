from copy import deepcopy
import numpy as np
from multiprocessing import Process, Queue
from tqdm import tqdm

from .exceptions import TimelimitError
from .samplers.utils import SAMPLERS_MAP

import warnings
warnings.filterwarnings("ignore")


class Optimizer(object):
    """Black-box Optimizer

    Black box optimization is a method to find optimal parameteres by
    observing the response of an objective function wrt each parameters
    without defining models, e.g., hyperparameter optimization.

    Parameters
    ----------
    score_func : function
        Takes dictionary as input and returns scalar score.
    space : list(dict)
        Each element define name search space as a dictionary.
    sampler: str
        The name of sample to use: 'grid', 'random', and 'bayes'
    init_X: array-like(float), shape=(n_samples, n_dim)
        The list of parameters to initizlie sampler
    init_y: array-like(float), shape(n_samples,)
        The list of score of init_X
    timeout: int, optional
        If specified, it terminates score evaluation after
        timeout seconds has passed.
    kwargs:
        These parameteres are sent to sampler object

    Here is the samples of how to define score_func and space:

    from sklearn.svm import SVC
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import StandardScaler


    data, target = make_classification(n_samples=2500,
                                       n_features=45,
                                       n_informative=5,
                                       n_redundant=5)

    space = [
        {'name': 'C', 'domain': (1e-8, 1e5), 'type': 'continuous', 'scale': 'log'},
        {'name': 'gamma', 'domain': (1e-8, 1e5), 'type': 'continuous', 'scale': 'log'},
        {'name': 'kernel', 'domain': 'rbf', 'type': 'fixed'}
    ]

    def score_func(params):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
        train_idx, test_idx = list(splitter.split(data, target))[0]
        train_data = data[train_idx]
        train_target = target[train_idx]
        clf = SVC(**params)
        clf.fit(train_data, train_target)

        pred = clf.predict(data[test_idx])
        true_y = target[test_idx]
        score = accuracy_score(true_y, pred)
        return -score
    """

    def __init__(self, score_func, space,
                 sampler="random", init_X=None, init_y=None,
                 maximize=False, timeout=None, **kwargs):
        self._score_func = score_func
        self._space_conf = space
        self._maximize = maximize
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
        self.sampler = SAMPLERS_MAP[sampler](self._nonfixed_conf,
                                             init_X, init_y, **kwargs)

    def search(self, return_full=False, num_iter=10, is_display=True,
               *args, **kwargs):
        """Find optimal set of parameters

        Parameters
        ----------
        return_full: bool (default False)
            If True, return all of search results
            If False, return only optimal set of paramters and its score
        num_iter: int (default 10)
            How many time to try
        is_display: bool (default)
            If True, show the progress bar

        Returns
        -------
        If return_full == True:
            Xs: list(dict)
            ys: list(float)
        If return_fulll == False
            best_X: dict
            ys: float
        """
        if is_display:
            iteration = tqdm(range(num_iter))
        else:
            iteration = range(num_iter)
        for i in iteration:
            Xs = self.sampler.sample(*args, **kwargs)
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
            self.sampler.update(sucess_Xs, ys)
        Xs, ys = self.sampler.data
        best_idx = np.argmin(ys)
        # Default is minimization
        if self._maximize:
            ys = -ys
        # Update with  fixed parameters
        fixed_params = deepcopy(self.fixed_params)
        for X in Xs:
            X.update(fixed_params)
        if return_full:
            return Xs, ys
        else:
            best_X = Xs[best_idx]
            best_y = ys[best_idx]
            return best_X, best_y

    def score_func(self, X, *args, **kwargs):
        fixed_params = deepcopy(self.fixed_params)
        X = deepcopy(X)
        X.update(fixed_params)

        def record(que):
            try:
                score = self._score_func(X, *args, **kwargs)
                # Default is minimization
                if self._maximize:
                    score = -score
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

    @property
    def results(self):
        best_X = []
        best_y = []
        X, y = self.sampler.data
        for i in range(len(y)):
            idx = np.argmin(y[:i + 1])
            best_X.append(X[idx])
            best_y.append(y[idx])
        best_y = np.array(best_y)
        if self._maximize:
            best_y = -best_y
        return best_X, best_y

    @property
    def best_results(self):
        X, y = self.sampler.data
        idx = np.argmin(y)
        best_X = X[idx]
        best_y = y[idx]
        if self._maximize:
            best_y = -best_y
        return best_X, best_y
