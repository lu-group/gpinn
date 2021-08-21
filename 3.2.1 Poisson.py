from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import deepxde as dde
from deepxde.backend import tf

# PINN
def PINNpde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    f = 8 * tf.sin(8 * x)
    for i in range(1, 5):
        f += i * tf.sin(i * x)
    return -dy_xx - f


def func(x):
    sol = x + 1 / 8 * np.sin(8 * x)
    for i in range(1, 5):
        sol += 1 / i * np.sin(i * x)
    return sol


geom = dde.geometry.Interval(0, np.pi)

data = dde.data.PDE(geom, PINNpde, [], 15, 0, "uniform", solution=func, num_test=100)

layer_size = [1] + [20] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)


def output_transform(x, y):
    return x + tf.math.tanh(x) * tf.math.tanh(np.pi - x) * y


net.apply_output_transform(output_transform)

PINNmodel = dde.Model(data, net)
PINNmodel.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = PINNmodel.train(epochs=20000)

dde.saveplot(losshistory, train_state, issave=True, isplot=False)

# gPINN
def gPINNpde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    dy_xxx = dde.grad.jacobian(dy_xx, x)

    f = 8 * tf.sin(8 * x)
    for i in range(1, 5):
        f += i * tf.sin(i * x)
    df_x = (
        tf.cos(x)
        + 4 * tf.cos(2 * x)
        + 9 * tf.cos(3 * x)
        + 16 * tf.cos(4 * x)
        + 64 * tf.cos(8 * x)
    )

    return [-dy_xx - f, -dy_xxx - df_x]


geom = dde.geometry.Interval(0, np.pi)

data = dde.data.PDE(geom, gPINNpde, [], 15, 0, "uniform", solution=func, num_test=100)

layer_size = [1] + [20] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)


def output_transform(x, y):
    return x + tf.math.tanh(x) * tf.math.tanh(np.pi - x) * y


net.apply_output_transform(output_transform)

gPINNmodel = dde.Model(data, net)
gPINNmodel.compile(
    "adam", lr=0.001, metrics=["l2 relative error"], loss_weights=[1, 0.01]
)
losshistory, train_state = gPINNmodel.train(epochs=20000)

dde.saveplot(losshistory, train_state, issave=True, isplot=False)

# plots

x = geom.uniform_points(1000)

plt.figure()
plt.plot(x, func(x), label="Exact", color="black")
plt.plot(x, PINNmodel.predict(x), label="PINN", color="blue", linestyle="dashed")
plt.plot(
    x, gPINNmodel.predict(x), label="gPINN, w = 0.01", color="red", linestyle="dashed"
)
plt.legend(frameon=False)

x = geom.uniform_points(15, boundary=False)
plt.plot(x, func(x), color="black", marker="o", linestyle="none")
plt.xlabel("x")
plt.ylabel("u")



x = geom.uniform_points(1000)
def du_x(x):
    return 1 + np.cos(x) + np.cos(2 * x) + np.cos(3 * x) + np.cos(4 * x) + np.cos(8 * x)

plt.figure()
plt.plot(x, du_x(x), label="Exact", color="black")
plt.plot(
    x,
    PINNmodel.predict(x, operator=lambda x, y: dde.grad.jacobian(y, x)),
    label="PINN",
    color="blue",
    linestyle="dashed",
)
plt.plot(
    x,
    gPINNmodel.predict(x, operator=lambda x, y: dde.grad.jacobian(y, x)),
    label="gPINN, w = 0.01",
    color="red",
    linestyle="dashed",
)

x = geom.uniform_points(15, boundary=False)
plt.plot(x, du_x(x), color="black", marker="o", linestyle="none")

plt.legend(frameon=False)
plt.xlabel("x")
plt.ylabel("u'")

plt.show()
