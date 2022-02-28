from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat

import deepxde as dde
from deepxde.backend import tf


def gen_testdata():
    data = loadmat("usol_D_0.001_k_5.mat")

    t = data["t"]

    x = data["x"]

    u = data["u"]

    dt = dx = 0.01

    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = u.flatten()[:, None]
    return X, y


# gPINN + RAR
def output_transform(x, y):
    x_in = x[:, 0:1]
    t_in = x[:, 1:2]
    return t_in * (1 + x_in) * (1 - x_in) * y + tf.square(x_in) * tf.cos(np.pi * x_in)


def gPINNpde(x, y):
    u = y
    du_xx = dde.grad.hessian(y, x, i=0, j=0)
    du_t = dde.grad.jacobian(y, x, j=1)

    du_tx = dde.grad.hessian(y, x, i=0, j=1)
    du_xxx = dde.grad.jacobian(du_xx, x, j=0)
    du_x = dde.grad.jacobian(y, x, j=0)

    du_tt = dde.grad.hessian(y, x, i=1, j=1)
    du_xxt = dde.grad.jacobian(du_xx, x, j=1)

    return [
        du_t - 0.001 * du_xx + 5 * (u ** 3 - u),
        du_tx - 0.001 * du_xxx + 5 * (3 * u ** 2 * du_x - du_x),
        du_tt - 0.001 * du_xxt + 5 * (3 * u ** 2 * du_t - du_t),
    ]


geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

data = dde.data.TimePDE(geomtime, gPINNpde, [], num_domain=500)
net = dde.maps.FNN([2] + [64] * 4 + [1], "tanh", "Glorot normal")
net.apply_output_transform(output_transform)
gPINNRARmodel = dde.Model(data, net)

gPINNRARmodel.compile("adam", lr=1.0e-3, loss_weights=[1, 0.0001, 0.0001])
losshistory, train_state = gPINNRARmodel.train(epochs=20000)
gPINNRARmodel.compile("L-BFGS-B", loss_weights=[1, 0.0001, 0.0001])
losshistory, train_state = gPINNRARmodel.train()

x_true, y_true = gen_testdata()
y_pred = gPINNRARmodel.predict(x_true)
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))


for i in range(100):
    X = geomtime.random_points(100000)
    err_eq = np.abs(gPINNRARmodel.predict(X, operator=gPINNpde))[0]

    err = np.mean(err_eq)
    print("Mean residual: %.3e" % (err))

    err_eq = torch.tensor(err_eq)
    x_ids = torch.topk(err_eq, 30, dim=0)[1].numpy()

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
