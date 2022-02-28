from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import deepxde as dde
from deepxde.backend import tf


def gen_testdata():
    data = np.load("Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y


# PINN + RAR
def output_transform(x, y):
    x_in = x[:, 0:1]
    t_in = x[:, 1:2]

    return (1 - x_in) * (1 + x_in) * (1 - tf.exp(-t_in)) * y - tf.sin(np.pi * x_in)


def PINNpde(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return [dy_t + y * dy_x - 0.01 / np.pi * dy_xx]


geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

data = dde.data.TimePDE(geomtime, PINNpde, [], num_domain=1500)
net = dde.maps.FNN([2] + [32] * 3 + [1], "tanh", "Glorot normal")
net.apply_output_transform(output_transform)

PINNRARmodel = dde.Model(data, net)

PINNRARmodel.compile("adam", lr=1e-3)
losshistory, train_state = PINNRARmodel.train(epochs=20000)
PINNRARmodel.compile("L-BFGS-B")
losshistory, train_state = PINNRARmodel.train()


x_true, y_true = gen_testdata()
y_pred = PINNRARmodel.predict(x_true)
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))


for i in range(40):
    X = geomtime.random_points(100000)
    f = PINNRARmodel.predict(X, operator=PINNpde)[0]
    err_eq = np.absolute(f)
    err = np.mean(err_eq)
    print("Mean residual: %.3e" % (err))

    err_eq = torch.tensor(err_eq)
    x_ids = torch.topk(err_eq, 10, dim=0)[1].numpy()

    for elem in x_ids:
        print("Adding new point:", X[elem], "\n")
        data.add_anchors(X[elem])
    early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-4, patience=2000)
    PINNRARmodel.compile("adam", lr=1e-3)

    losshistory, train_state = PINNRARmodel.train(
        epochs=10000, disregard_previous_best=True, callbacks=[early_stopping]
    )

    PINNRARmodel.compile("L-BFGS-B")
    losshistory, train_state = PINNRARmodel.train()

    y_pred = PINNRARmodel.predict(x_true)
    print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# gPINN + RAR
def output_transform(x, y):
    x_in = x[:, 0:1]
    t_in = x[:, 1:2]

    return (1 - x_in) * (1 + x_in) * (1 - tf.exp(-t_in)) * y - tf.sin(np.pi * x_in)


def pde(x, y):
    dy_x = dde.grad.jacobian(y, x, j=0)
    dy_t = dde.grad.jacobian(y, x, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)

    dy_tx = dde.grad.hessian(y, x, i=0, j=1)
    dy_xxx = dde.grad.jacobian(dy_xx, x, j=0)

    dy_tt = dde.grad.hessian(y, x, i=1, j=1)
    dy_xxt = dde.grad.jacobian(dy_xx, x, j=1)
    return [
        dy_t + y * dy_x - 0.01 / np.pi * dy_xx,
        dy_tx + (dy_x * dy_x + y * dy_xx) - 0.01 / np.pi * dy_xxx,
        dy_tt + dy_t * dy_x + y * dy_tx - 0.01 / np.pi * dy_xxt,
    ]


geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.IC(
    geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial
)

data = dde.data.TimePDE(
    geomtime, pde, [], num_domain=1500, num_boundary=0, num_initial=0
)
net = dde.maps.FNN([2] + [32] * 3 + [1], "tanh", "Glorot normal")
net.apply_output_transform(output_transform)
gPINNRARmodel = dde.Model(data, net)

gPINNRARmodel.compile("adam", lr=1.0e-3, loss_weights=[1, 0.0001, 0.0001])
losshistory, train_state = gPINNRARmodel.train(epochs=20000)
gPINNRARmodel.compile("L-BFGS-B", loss_weights=[1, 0.0001, 0.0001])
losshistory, train_state = gPINNRARmodel.train()

x_true, y_true = gen_testdata()
y_pred = gPINNRARmodel.predict(x_true)
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))


for i in range(40):
    X = geomtime.random_points(100000)
    err_eq = np.abs(gPINNRARmodel.predict(X, operator=pde))[0]

    err = np.mean(err_eq)
    print("Mean residual: %.3e" % (err))

    err_eq = torch.tensor(err_eq)
    x_ids = torch.topk(err_eq, 10, dim=0)[1].numpy()

    for elem in x_ids:
        print("Adding new point:", X[elem], "\n")
        data.add_anchors(X[elem])
    early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-4, patience=2000)
    gPINNRARmodel.compile("adam", lr=1e-3, loss_weights=[1, 0.0001, 0.0001])
    losshistory, train_state = gPINNRARmodel.train(
        epochs=10000, disregard_previous_best=True, callbacks=[early_stopping]
    )
    gPINNRARmodel.compile("L-BFGS-B", loss_weights=[1, 0.0001, 0.0001])
    losshistory, train_state = gPINNRARmodel.train()

    X, y_true = gen_testdata()
    y_pred = gPINNRARmodel.predict(X)
    print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
