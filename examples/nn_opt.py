from sklearn.preprocessing import OneHotEncoder
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.layers import Activation, Reshape
from keras.optimizers import Adam, Adadelta, SGD, RMSprop
from keras.regularizers import l2
import matplotlib.pyplot as plt

from bboptimizer import Optimizer

# Fetch MNIST dataset
mnist = tf.contrib.learn.datasets.load_dataset("mnist")


train = mnist.train
X = train.images
train_X = X
train_y = np.expand_dims(train.labels, -1)
train_y = OneHotEncoder().fit_transform(train_y)

valid = mnist.validation
X = valid.images
valid_X = X
valid_y = np.expand_dims(valid.labels, -1)
valid_y = OneHotEncoder().fit_transform(valid_y)


def get_optimzier(name, **kwargs):
    if name == "rmsprop":
        return RMSprop(**kwargs)
    elif name == "adam":
        return Adam(**kwargs)
    elif name == "sgd":
        return SGD(**kwargs)
    elif name == "adadelta":
        return Adadelta(**kwargs)
    else:
        raise ValueError(name)


def construct_NN(params):
    model = Sequential()
    model.add(Reshape((784,), input_shape=(784,)))

    def update_model(_model, _params, name):
        _model.add(Dropout(_params[name + "_drop_rate"]))
        _model.add(Dense(units=_params[name + "_num_units"],
                    activation=None,
                    kernel_regularizer=l2(_params[name + "_w_reg"])))
        if _params[name + "_is_batch"]:
            _model.add(BatchNormalization())
        if _params[name + "_activation"] is not None:
            _model.add(Activation(_params[name + "_activation"]))
        return _model

    # Add input layer
    model = update_model(model, params, "input")
    # Add hidden layer
    for i in range(params["num_hidden_layers"]):
        model = update_model(model, params, "hidden")
    # Add output layer
    model = update_model(model, params, "output")
    optimizer = get_optimzier(params["optimizer"],
                              lr=params["learning_rate"])
    model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model


def score_func(params):
    # print("parameters", params)
    model = construct_NN(params)
    model.fit(train_X, train_y,
              epochs=params["epochs"],
              batch_size=params["batch_size"], verbose=1)
    # print("###################", model.metrics_names)
    score = model.evaluate(valid_X, valid_y,
                  batch_size=params["batch_size"])
    idx = model.metrics_names.index("acc")
    score = score[idx]
    print(params, score)
    return score

params_conf = [
    {"name": "num_hidden_layers", "type": "integer",
     "domain": (0, 5)},
    {"name": "batch_size", "type": "integer",
     "domain": (16, 128), "scale": "log"},
    {"name": "learning_rate", "type": "continuous",
     "domain": (1e-5, 1e-1), "scale": "log"},
    {"name": "epochs", "type": "integer",
     "domain": (10, 250), "scale": "log"},
    {"name": "optimizer", "type": "categorical",
     "domain": ("rmsprop", "sgd", "adam", "adadelta")},

    {"name": "input_drop_rate", "type": "continuous",
     "domain": (0, 0.5)},
    {"name": "input_num_units", "type": "integer",
     "domain": (32, 512), "scale": "log"},
    {"name": "input_w_reg", "type": "continuous",
     "domain": (1e-10, 1e-1), "scale": "log"},
    {"name": "input_is_batch", "type": "categorical",
     "domain": (True, False)},
    {"name": "input_activation", "type": "categorical",
     "domain": ("relu", "sigmoid", "tanh")},

    {"name": "hidden_drop_rate", "type": "continuous",
     "domain": (0, 0.75)},
    {"name": "hidden_num_units", "type": "integer",
     "domain": (32, 512), "scale": "log"},
    {"name": "hidden_w_reg", "type": "continuous",
     "domain": (1e-10, 1e-1), "scale": "log"},
    {"name": "hidden_is_batch", "type": "categorical",
     "domain": (True, False)},
    {"name": "hidden_activation", "type": "categorical",
     "domain": ("relu", "sigmoid", "tanh")},

    {"name": "output_drop_rate", "type": "continuous",
     "domain": (0, 0.5)},
    {"name": "output_num_units", "type": "fixed",
     "domain": 10},
    {"name": "output_w_reg", "type": "continuous",
     "domain": (1e-10, 1e-1), "scale": "log"},
    {"name": "output_is_batch", "type": "categorical",
     "domain": (True, False)},
    {"name": "output_activation", "type": "fixed",
     "domain": "softmax"},

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

    # Plot results
    plt.figure(figsize=(20, 10))
    X = np.arange(1, len(bayes_opt.results[1]) + 1)
    plt.plot(X, bayes_opt.results[1], color="b", label="bayes")
    plt.plot(X, random_opt.results[1], color="g", label="random")

    plt.scatter(X, bayes_opt.results[1], color="b")
    plt.scatter(X, random_opt.results[1], color="g")

    plt.xlabel("the number of trials", fontsize=30)
    plt.ylabel("score", fontsize=30)
    plt.title("Neural Network Hyperparameter Optimization", fontsize=50)

    plt.ylim(0.96, 1.0)

    plt.legend(fontsize=20)
    plt.savefig("hyper_nn_opt.jpg")
