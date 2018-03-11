import random
import numpy as np
import matplotlib.pyplot as plt

from bboptimizer import Optimizer

map_func = dict(linear=lambda x: x, square=lambda x: x**2, sin=np.sin)

def score_func(x):
    score = np.sin(x["x2_1"]) + map_func[x["x4"]](x["x1"]) + map_func[x["x4"]](x["x2_2"])
    return score


params_conf = [
    {"name": "x1", "domain": (.1, 5), "type": "continuous",
     "num_grid": 5, "scale": "log"},
    {"name": "x2", "domain": (-5, 5), "type": "continuous",
     "num_grid": 5, "dimensionality": 2},
    {"name": "x4", "domain": ("linear", "sin", "square"),
     "type": "categorical"},
]


if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    bayes_opt = Optimizer(score_func, params_conf, sampler="bayes", r_min=10, maximize=True)
    print("****************************")
    print("bayes")
    print(bayes_opt.search(num_iter=50))

    print("****************************")
    print("random")
    np.random.seed(0)
    random.seed(0)
    random_opt = Optimizer(score_func, params_conf, sampler="random", maximize=True)
    random_opt.search(num_iter=50)

    print("****************************")
    print("grid")
    np.random.seed(0)
    random.seed(0)
    grid_opt = Optimizer(score_func, params_conf, sampler="grid", num_grid=3, maximize=True)
    grid_opt.search(num_iter=50)

    # Plot results
    plt.figure(figsize=(20, 10))
    X = np.arange(1, len(bayes_opt.results[1]) + 1)
    plt.plot(X, bayes_opt.results[1], color="b", label="bayes")
    plt.plot(X, random_opt.results[1], color="g", label="random")
    plt.plot(X, grid_opt.results[1], color="y", label="grid")

    plt.scatter(X, bayes_opt.results[1], color="b")
    plt.scatter(X, random_opt.results[1], color="g")
    plt.scatter(X, grid_opt.results[1], color="y")

    plt.xlabel("the number of trials", fontsize=30)
    plt.ylabel("score", fontsize=30)
    plt.title("Optimization results", fontsize=50)

    plt.legend(fontsize=20)
    plt.savefig("toy_model_opt.jpg")
