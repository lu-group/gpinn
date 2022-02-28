from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import deepxde as dde

from deepxde.backend import tf


def func(x):
    return -(1.4 - 3 * x) * np.sin(18 * x)


# NN
def NNfunc(x, y):
    return y + (1.4 - 3 * x) * tf.sin(18 * x)


geom = dde.geometry.Interval(0, 1)
data = dde.data.PDE(geom, NNfunc, [], 13, 2, "uniform", solution=func, num_test=100)

activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN([1] + [20] * 3 + [1], activation, initializer)

NNmodel = dde.Model(data, net)
NNmodel.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = NNmodel.train(epochs=10000)

dde.saveplot(losshistory, train_state, issave=True, isplot=False)

# gNN
def gNNfunc(x, y):
    dy_x = dde.grad.jacobian(y, x)

    return [
        y + (1.4 - 3 * x) * tf.sin(18 * x),
        dy_x + 18 * (1.4 - 3 * x) * tf.cos(18 * x) - 3 * tf.sin(18 * x),
    ]


geom = dde.geometry.Interval(0, 1)
data = dde.data.PDE(geom, gNNfunc, [], 13, 2, "uniform", solution=func, num_test=100)

activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN([1] + [20] * 3 + [1], activation, initializer)

gNNmodel = dde.Model(data, net)
gNNmodel.compile(
    "adam", lr=0.001, metrics=["l2 relative error"], loss_weights=[1, 0.01]
)
losshistory, train_state = gNNmodel.train(epochs=10000)

dde.saveplot(losshistory, train_state, issave=True, isplot=False)

# plots

plt.rcParams.update({"font.size": 16})

x = geom.uniform_points(1000)
plt.figure()
plt.plot(x, func(x), label="Exact", color="black")
plt.plot(x, NNmodel.predict(x), label="NN", color="blue", linestyle="dashed")
plt.plot(x, gNNmodel.predict(x), label="gNN", color="red", linestyle="dashed")

x = geom.uniform_points(15)
plt.plot(x, func(x), color="black", marker="o", linestyle="none")

plt.xlabel("x")
plt.ylabel("u")

plt.legend(frameon=False)


def du_x(x):
    return 3 * np.sin(18 * x) + 18 * (3 * x - 1.4) * np.cos(18 * x)


x = geom.uniform_points(1000)
plt.figure()
plt.plot(x, du_x(x), label="Exact", color="black")
plt.plot(
    x,
    NNmodel.predict(x, operator=lambda x, y: dde.grad.jacobian(y, x)),
    label="NN",
    color="blue",
    linestyle="dashed",
)
plt.plot(
    x,
    gNNmodel.predict(x, operator=lambda x, y: dde.grad.jacobian(y, x)),
    label="gNN",
    color="red",
    linestyle="dashed",
)

x = geom.uniform_points(15)
plt.plot(x, du_x(x), color="black", marker="o", linestyle="none")
plt.xlabel("x")
plt.ylabel("u'")

plt.legend(frameon=False)

plt.show()
