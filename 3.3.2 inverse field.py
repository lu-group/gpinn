from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import sys
import re

import deepxde as dde
from deepxde.backend import tf


def k(x):
    return 0.1 + np.exp(-0.5 * (x - 0.5) ** 2 / 0.15 ** 2)


def fun(x, y):
    return np.vstack((y[1], 100 * (k(x) * y[0] + np.sin(2 * np.pi * x))))


def bc(ya, yb):
    return np.array([ya[0], yb[0]])


a = np.linspace(0, 1, 1000)
b = np.zeros((2, a.size))

res = solve_bvp(fun, bc, a, b)


def sol(x):
    return res.sol(x)[0]


def du(x):
    return res.sol(x)[1]


# PINN
l = 0.01


def gen_traindata(num):
    xvals = np.linspace(0, 1, num)
    yvals = sol(xvals)

    return np.reshape(xvals, (-1, 1)), np.reshape(yvals, (-1, 1))


def output_transform(x, y):
    return tf.concat(
        (tf.math.tanh(x) * tf.math.tanh(1 - x) * y[:, 0:1], y[:, 1:2]), axis=1
    )


geom = dde.geometry.Interval(0, 1)

ob_x, ob_u = gen_traindata(8)
observe_u = dde.PointSetBC(ob_x, ob_u, component=0)
bc = dde.DirichletBC(geom, sol, lambda _, on_boundary: on_boundary, component=0)


def pde(x, y):
    u = y[:, 0:1]
    k = y[:, 1:2]

    du_xx = dde.grad.hessian(y, x, component=0)

    return l * du_xx - k * u - tf.sin(2 * np.pi * x)


data = dde.data.PDE(
    geom,
    pde,
    bcs=[bc, observe_u],
    num_domain=8,
    num_boundary=2,
    train_distribution="uniform",
    num_test=1000,
)
net = dde.maps.PFNN([1, [20, 20], [20, 20], [20, 20], 2], "tanh", "Glorot uniform")
PINNmodel = dde.Model(data, net)
PINNmodel.compile("adam", lr=0.0001, metrics=[])
losshistory, train_state = PINNmodel.train(epochs=200000, callbacks=[])
dde.saveplot(losshistory, train_state, issave=True, isplot=False)

# gPINN
l = 0.01


def gen_traindata(num):
    xvals = np.linspace(0, 1, num)
    yvals = sol(xvals)

    return np.reshape(xvals, (-1, 1)), np.reshape(yvals, (-1, 1))


def output_transform(x, y):
    return tf.concat((x * (1 - x) * y[:, 0:1], y[:, 1:2]), axis=1)


geom = dde.geometry.Interval(0, 1)

ob_x, ob_u = gen_traindata(8)
observe_u = dde.PointSetBC(ob_x, ob_u, component=0)
bc = dde.DirichletBC(geom, sol, lambda _, on_boundary: on_boundary, component=0)


def pde(x, y):
    u = y[:, 0:1]
    k = y[:, 1:2]

    du_x = dde.grad.jacobian(y, x, i=0)
    du_xx = dde.grad.hessian(y, x, component=0)
    du_xxx = dde.grad.jacobian(du_xx, x)

    dk_x = dde.grad.jacobian(y, x, i=1)

    return [
        l * du_xx - k * u - tf.sin(2 * np.pi * x),
        l * du_xxx - k * du_x - u * dk_x - 2 * np.pi * tf.cos(2 * np.pi * x),
    ]


data = dde.data.PDE(
    geom,
    pde,
    bcs=[bc, observe_u],
    num_domain=8,
    num_boundary=2,
    train_distribution="uniform",
    num_test=1000,
)
net = dde.maps.PFNN([1, [20, 20], [20, 20], [20, 20], 2], "tanh", "Glorot uniform")
gPINNmodel = dde.Model(data, net)
gPINNmodel.compile("adam", lr=0.0001, metrics=[], loss_weights=[1, 0.01, 1, 1])
losshistory, train_state = gPINNmodel.train(epochs=200000, callbacks=[])
dde.saveplot(losshistory, train_state, issave=True, isplot=False)

# plots
plt.rcParams.update({"font.size": 16})

x = geom.uniform_points(1000)

plt.figure()
plt.plot(x, sol(x), label="Exact", color="black")
plt.plot(x, PINNmodel.predict(x)[:, 0], label="PINN", linestyle="dashed", color="blue")
plt.plot(x, gPINNmodel.predict(x)[:, 0], label="gPINN", linestyle="dashed", color="red")

x = geom.uniform_points(8)
plt.plot(x, sol(x), label="Observed", color="black", marker="s", linestyle="none")
plt.legend(frameon=False)
plt.ylabel("u")
plt.xlabel("x")


x = geom.uniform_points(1000)
plt.figure()
plt.plot(x, k(x), label="Exact", color="black")
plt.plot(x, PINNmodel.predict(x)[:, 1], label="PINN", linestyle="dashed", color="blue")
plt.plot(x, gPINNmodel.predict(x)[:, 1], label="gPINN", linestyle="dashed", color="red")
plt.legend(frameon=False)
plt.ylabel("K")
plt.xlabel("x")

plt.figure()
plt.plot(x, du(x), label="Exact", color="black")
plt.plot(
    x,
    PINNmodel.predict(x, operator=lambda x, y: dde.grad.jacobian(y, x)),
    label="PINN",
    linestyle="dashed",
    color="blue",
)
plt.plot(
    x,
    gPINNmodel.predict(x, operator=lambda x, y: dde.grad.jacobian(y, x)),
    label="gPINN",
    linestyle="dashed",
    color="red",
)
plt.legend(frameon=False)
plt.ylabel("u'")
plt.xlabel("x")

plt.show()
