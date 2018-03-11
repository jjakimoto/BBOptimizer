Black Box Optimizer
===================

General black-box optimization mainly aiming for optimizing hyperparameters of Machine Learning algorithms. Defining function to be optimized, search space, and method gives you optimal set of parameters.

[![license](https://img.shields.io/badge/licence-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Current implementations
=======================
- Grid Search
- Random Search [[1]](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)
- Bayesian Search [[2]](https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf)



Getting started
===============

### Installation
```bash
    git clone https://github.com/jjakimoto/BBOptimizer
    cd BBOptimizer
    python setup.py install
```

### Dependencies:
------------------
  - scikit-learn
  - numpy
  - scipy
  - GPy

You can install dependencies by running:
```bash
pip install -r requirements.txt
```


### Example
Let's see how to use function while going through optimizing hyperparameters of support vector machine from scikit-learn (SVC).

First, we define the type of parameters from four kinds: `continuous`, `integer`, `categorical`, and `fixed`. `fixed` parameters will just be added to additional parameters for score function and not be included in optimization. Here is the example:
```python
space_conf = [
    {'name': 'C', 'domain': (1e-8, 1e5), 'type': 'continuous', 'scale': 'log'},
    {'name': 'gamma', 'domain': (1e-8, 1e5), 'type': 'continuous', 'scale': 'log'},
    {'name': 'kernel', 'domain': 'rbf', 'type': 'fixed'}
]
```
If you define `{'scale': 'log'}`, the parameter is optimized and sampled after transformed into `10 ** x`. We also have dimensionality option. If you add `{'dimensionality': N}`, for variable `x`, you would use `x_1`, `x_2`, ..., `x_N` with the same search  space.

Next we need to define a problem to be optimized. Score function has to get argument in the dictionary format and then return scalar values (currently support only single output case). Here is the example of accuracy score function of support vector machine classifier.

```python
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


data, target = make_classification(n_samples=2500,
                                   n_features=45,
                                   n_informative=5,
                                   n_redundant=5)

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
```

Then, we can execute optimization in the following way:
```python
from bboptimizer import Optimizer

opt = Optimizer(score_func, params_conf, sampler="bayes",
                r_min=10, backend="gpy")
opt.search(num_iter=20)
```

Here is the list of `Optimizer`'s `__init__` arguments
```text
score_func : function
    Takes dictionary as input and returns scalar score.
space: list(dict)
    Define search space. Each element has to the following key
    values: 'name', 'type', and 'domain' (,'num_grid' is optional).
sampler: str
    The name of sample to use: 'grid', 'random', and 'bayes'
init_X: array-like(float), shape=(n_samples, n_dim)
    The list of parameters to initialize sampler
init_y: array-like(float), shape(n_samples,)
    The list of score of init_X
timeout: int, optional
    If specified, it terminates score evaluation after
    timeout seconds has passed.
kwargs:
    These parameters are sent to sampler object
```

`sampler` and `backend` parameters mainly define your optimization method.

`sampler` parameter defines which optimization in use. It has the following options: `grid`, `random`, and `bayes`.

`backend` parameter just determines which gaussian process to use for bayesian optimization. It has to be either of `gpy` or `sklearn`. Default is `gpy`.


### References
- [1] [Random Search for Hyper-Parameter Optimization](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)
- [2] [Practical Bayesian Optimization of Machine Learning Algorithms](https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf)
