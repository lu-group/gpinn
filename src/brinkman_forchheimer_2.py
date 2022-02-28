from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import re

import deepxde as dde
from deepxde.backend import tf

# PINN
g = 1
v = 1e-3
e = 0.4
H = 1


def sol(x):
    r = (v * e / (1e-3 * 1e-3)) ** (0.5)
    return g * 1e-3 / v * (1 - np.cosh(r * (x - H / 2)) / np.cosh(r * H / 2))


def gen_traindata(num):
    xvals = np.linspace(1 / (num + 1), 1, num, endpoint=False)
    yvals = sol(xvals)

    return np.reshape(xvals, (-1, 1)), np.reshape(yvals, (-1, 1))


def output_transform(x, y):
    return tf.math.tanh(x) * tf.math.tanh(1 - x) * y


def du(x):
    r = (v * e / (1e-3 * K)) ** (0.5)
    return g * 1e-3 / v * (-r * np.sinh(r * (x - H / 2)) / np.cosh(r * H / 2))


geom = dde.geometry.Interval(0, 1)

ob_x, ob_u = gen_traindata(5)
observe_u = dde.PointSetBC(ob_x, ob_u, component=0)

v_e = tf.math.softplus(tf.Variable(0, trainable=True, dtype=tf.float32)) * 0.1
K = tf.math.softplus(tf.Variable(0, trainable=True, dtype=tf.float32)) * 0.1


def pde(x, y):
    u = y
    du_xx = dde.grad.hessian(y, x)

    return -v_e / e * du_xx + v * u / K - g


data = dde.data.PDE(
    geom,
    pde,
    solution=sol,
    bcs=[observe_u],
    num_domain=10,
    num_boundary=0,
    train_distribution="uniform",
    num_test=1000,
)

net = dde.maps.FNN([1] + [20] * 3 + [1], "tanh", "Glorot uniform")
net.apply_output_transform(output_transform)
PINNmodel = dde.Model(data, net)
PINNmodel.compile("adam", lr=0.001, metrics=["l2 relative error"])
variable = dde.callbacks.VariableValue([v_e, K], period=200, filename="variables1.dat")
losshistory, train_state = PINNmodel.train(epochs=50000, callbacks=[variable])
dde.saveplot(losshistory, train_state, issave=True, isplot=False)

# gPINN
g = 1
v = 1e-3
e = 0.4
H = 1


def sol(x):
    r = (v * e / (1e-3 * 1e-3)) ** (0.5)
    return g * 1e-3 / v * (1 - np.cosh(r * (x - H / 2)) / np.cosh(r * H / 2))


def gen_traindata(num):
    xvals = np.linspace(1 / (num + 1), 1, num, endpoint=False)
    yvals = sol(xvals)

    return np.reshape(xvals, (-1, 1)), np.reshape(yvals, (-1, 1))


def output_transform(x, y):
    return tf.math.tanh(x) * tf.math.tanh(1 - x) * y


def du(x):
    r = (v * e / (1e-3 * K)) ** (0.5)
    return g * K / v * (-r * np.sinh(r * (x - H / 2)) / np.cosh(r * H / 2))


geom = dde.geometry.Interval(0, 1)

ob_x, ob_u = gen_traindata(5)
observe_u = dde.PointSetBC(ob_x, ob_u, component=0)

v_e = tf.math.softplus(tf.Variable(0, trainable=True, dtype=tf.float32)) * 0.1
K = tf.math.softplus(tf.Variable(0, trainable=True, dtype=tf.float32)) * 0.1


def pde(x, y):
    u = y
    du_x = dde.grad.jacobian(y, x)
    du_xx = dde.grad.hessian(y, x)
    du_xxx = dde.grad.jacobian(du_xx, x)

    return [-v_e / e * du_xx + v * u / K - g, -v_e / e * du_xxx + v / K * du_x]


data = dde.data.PDE(
    geom,
    pde,
    solution=sol,
    bcs=[observe_u],
    num_domain=10,
    num_boundary=0,
    train_distribution="uniform",
    num_test=1000,
)

net = dde.maps.FNN([1] + [20] * 3 + [1], "tanh", "Glorot uniform")
net.apply_output_transform(output_transform)
gPINNmodel = dde.Model(data, net)
gPINNmodel.compile(
    "adam", lr=0.001, metrics=["l2 relative error"], loss_weights=[1, 0.1, 1]
)
variable = dde.callbacks.VariableValue([v_e, K], period=200, filename="variables2.dat")
losshistory, train_state = gPINNmodel.train(epochs=50000, callbacks=[variable])
dde.saveplot(losshistory, train_state, issave=True, isplot=False)

# plots
plt.rcParams.update({"font.size": 16})

x = geom.uniform_points(1000)
yhat1 = PINNmodel.predict(x)
uhat1, v_ehat1 = yhat1[:, 0:1], yhat1[:, 0:1]

yhat2 = gPINNmodel.predict(x)
uhat2, v_ehat1 = yhat2[:, 0:1], yhat2[:, 0:1]

plt.figure()
plt.plot(x, sol(x), label="Exact", color="black")
plt.plot(x, uhat1, label="PINN", linestyle="dashed", color="blue")
plt.plot(x, uhat2, label="gPINN", linestyle="dashed", color="red")

x = geom.uniform_points(5, boundary=False)
plt.plot(x, sol(x), color="black", marker="s", label="Observed", linestyle="none")


plt.legend(frameon=False)
plt.xlabel("x")
plt.ylabel("u")
plt.savefig("e1", dpi=300, bbox_inches="tight")


lines = open("variables1.dat", "r").readlines()
v_ehat1 = np.array(
    [
        np.fromstring(
            min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
            sep=",",
        )
        for line in lines
    ]
)

lines = open("variables2.dat", "r").readlines()
v_ehat2 = np.array(
    [
        np.fromstring(
            min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
            sep=",",
        )
        for line in lines
    ]
)

l, c = v_ehat1.shape
v_etrue = 1e-3

plt.figure()
plt.plot(
    range(0, 200 * l, 200),
    np.ones(v_ehat1[:, 0].shape) * v_etrue,
    color="black",
    label="Exact",
)
plt.plot(range(0, 200 * l, 200), v_ehat1[:, 0], "b--", label="PINN")
plt.plot(range(0, 200 * l, 200), v_ehat2[:, 0], "r--", label="gPINN")

plt.xlabel("Epoch")
plt.yscale("log")
plt.ylim(ymax=1e-1)

plt.legend(frameon=False)
plt.ylabel(r"$\nu_e$")

plt.figure()
plt.plot(
    range(0, 200 * l, 200),
    np.ones(v_ehat1[:, 0].shape) * v_etrue,
    color="black",
    label="Exact",
)
plt.plot(range(0, 200 * l, 200), v_ehat1[:, 0], "b--", label="PINN")
plt.plot(range(0, 200 * l, 200), v_ehat2[:, 0], "r--", label="gPINN")

plt.xlabel("Epoch")
plt.yscale("log")
plt.ylim(ymax=1e-1)
plt.legend(frameon=False)
plt.ylabel(r"$K$")

plt.show()
