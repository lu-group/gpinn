from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import deepxde as dde
from deepxde.backend import tf

# PINN
def PINNpde(x, y):
    x_in = x[:, 0:1]
    t_in = x[:, 1:2]

    dy_t = dde.grad.jacobian(y, x, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    r = tf.exp(-t_in) * (
        3 * tf.sin(2 * x_in) / 2
        + 8 * tf.sin(3 * x_in) / 3
        + 15 * tf.sin(4 * x_in) / 4
        + 63 * tf.sin(8 * x_in) / 8
    )

    dy_tx = dde.grad.hessian(y, x, i=0, j=1)
    dy_xxx = dde.grad.jacobian(dy_xx, x, j=0)
    dr_x = tf.exp(-t_in) * (
        63 * tf.cos(8 * x_in)
        + 15 * tf.cos(4 * x_in)
        + 8 * tf.cos(3 * x_in)
        + 3 * tf.cos(2 * x_in)
    )

    dy_tt = dde.grad.hessian(y, x, i=1, j=1)
    dy_xxt = dde.grad.jacobian(dy_xx, x, j=1)
    dr_t = -r

    return [dy_t - dy_xx - r]


def solution(a):
    x, t = a[:, 0:1], a[:, 1:2]
    val = np.sin(8 * x) / 8
    for i in range(1, 5):
        val += np.sin(i * x) / i
    return np.exp(-t) * val


def icfunc(x):
    return (
        tf.sin(8 * x) / 8
        + tf.sin(1 * x) / 1
        + tf.sin(2 * x) / 2
        + tf.sin(3 * x) / 3
        + tf.sin(4 * x) / 4
    )


def output_transform(x, y):
    x_in = x[:, 0:1]
    t_in = x[:, 1:2]

    return (x_in - np.pi) * (x_in + np.pi) * (1 - tf.exp(-t_in)) * y + icfunc(x_in)


geom = dde.geometry.Interval(-np.pi, np.pi)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


def du_t(x):
    x_in, t_in = x[:, 0:1], x[:, 1:2]

    val = np.sin(8 * x_in) / 8
    for i in range(1, 5):
        val += np.sin(i * x_in) / i
    return -np.exp(-t_in) * val


def du_x(x):
    x_in, t_in = x[:, 0:1], x[:, 1:2]

    val = np.cos(8 * x_in)
    for i in range(1, 5):
        val += np.cos(i * x_in)
    return np.exp(-t_in) * val


data = dde.data.TimePDE(
    geomtime,
    PINNpde,
    [],
    num_domain=50,
    # train_distribution="uniform",
    solution=solution,
    num_test=10000,
)

layer_size = [2] + [20] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

net.apply_output_transform(output_transform)

PINNmodel = dde.Model(data, net)

PINNmodel.compile("adam", lr=0.0001, metrics=["l2 relative error"])

losshistory, train_state = PINNmodel.train(epochs=100000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)


# gPINN
def gPINNpde(x, y):
    x_in = x[:, 0:1]
    t_in = x[:, 1:2]

    dy_t = dde.grad.jacobian(y, x, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    r = tf.exp(-t_in) * (
        3 * tf.sin(2 * x_in) / 2
        + 8 * tf.sin(3 * x_in) / 3
        + 15 * tf.sin(4 * x_in) / 4
        + 63 * tf.sin(8 * x_in) / 8
    )

    dy_tx = dde.grad.hessian(y, x, i=0, j=1)
    dy_xxx = dde.grad.jacobian(dy_xx, x, j=0)
    dr_x = tf.exp(-t_in) * (
        63 * tf.cos(8 * x_in)
        + 15 * tf.cos(4 * x_in)
        + 8 * tf.cos(3 * x_in)
        + 3 * tf.cos(2 * x_in)
    )

    dy_tt = dde.grad.hessian(y, x, i=1, j=1)
    dy_xxt = dde.grad.jacobian(dy_xx, x, j=1)
    dr_t = -r

    return [dy_t - dy_xx - r, dy_tx - dy_xxx - dr_x, dy_tt - dy_xxt - dr_t]


def solution(a):
    x, t = a[:, 0:1], a[:, 1:2]
    val = np.sin(8 * x) / 8
    for i in range(1, 5):
        val += np.sin(i * x) / i
    return np.exp(-t) * val


def icfunc(x):
    return (
        tf.sin(8 * x) / 8
        + tf.sin(1 * x) / 1
        + tf.sin(2 * x) / 2
        + tf.sin(3 * x) / 3
        + tf.sin(4 * x) / 4
    )


def output_transform(x, y):
    x_in = x[:, 0:1]
    t_in = x[:, 1:2]

    return (x_in - np.pi) * (x_in + np.pi) * (1 - tf.exp(-t_in)) * y + icfunc(x_in)


geom = dde.geometry.Interval(-np.pi, np.pi)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


def du_t(x):
    x_in, t_in = x[:, 0:1], x[:, 1:2]

    val = np.sin(8 * x_in) / 8
    for i in range(1, 5):
        val += np.sin(i * x_in) / i
    return -np.exp(-t_in) * val


def du_x(x):
    x_in, t_in = x[:, 0:1], x[:, 1:2]

    val = np.cos(8 * x_in)
    for i in range(1, 5):
        val += np.cos(i * x_in)
    return np.exp(-t_in) * val


data = dde.data.TimePDE(
    geomtime,
    gPINNpde,
    [],
    num_domain=50,
    solution=solution,
    num_test=10000,
)

layer_size = [2] + [20] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

net.apply_output_transform(output_transform)

gPINNmodel = dde.Model(data, net)

gPINNmodel.compile(
    "adam", lr=0.0001, metrics=["l2 relative error"], loss_weights=[1, 0.1, 0.1]
)

losshistory, train_state = gPINNmodel.train(epochs=100000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# results
x = geomtime.random_points(50000)
print("L2 relative error of u:")
print("\tPINN:", dde.metrics.l2_relative_error(solution(x), PINNmodel.predict(x)))
print("\tgPINN:", dde.metrics.l2_relative_error(solution(x), gPINNmodel.predict(x)))

PINNresiduals = PINNmodel.predict(x, operator=PINNpde)[0]
gPINNresiduals = gPINNmodel.predict(x, operator=gPINNpde)[0]
print("Mean absolute PDE residual:")
print("\tPINN:", np.mean(abs(PINNresiduals)))
print("\tgPINN:", np.mean(abs(gPINNresiduals)))

print("L2 relative error of du_x")
print(
    "\tPINN:",
    dde.metrics.l2_relative_error(
        du_x(x), PINNmodel.predict(x, lambda x, y: dde.grad.jacobian(y, x, j=0))
    ),
)
print(
    "\tgPINN:",
    dde.metrics.l2_relative_error(
        du_x(x), gPINNmodel.predict(x, lambda x, y: dde.grad.jacobian(y, x, j=0))
    ),
)

print("L2 relative error of du_t")
print(
    "\tPINN:",
    dde.metrics.l2_relative_error(
        du_t(x), PINNmodel.predict(x, lambda x, y: dde.grad.jacobian(y, x, j=1))
    ),
)
print(
    "\tgPINN:",
    dde.metrics.l2_relative_error(
        du_t(x), gPINNmodel.predict(x, lambda x, y: dde.grad.jacobian(y, x, j=1))
    ),
)
