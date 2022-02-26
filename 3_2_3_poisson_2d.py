from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import deepxde as dde
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from deepxde.backend import tf

# PINN
a = 10


def f(x, y):
    u_xx = (
        16**a
        * a
        * (a * (1 - 2 * x) ** 2 - 2 * x**2 + 2 * x - 1)
        * ((x - 1) * x * (y - 1) * y) ** a
        / ((x - 1) ** 2 * x**2)
    )
    u_yy = (
        16**a
        * a
        * (a * (1 - 2 * y) ** 2 - 2 * y**2 + 2 * y - 1)
        * ((x - 1) * x * (y - 1) * y) ** a
        / ((y - 1) ** 2 * y**2)
    )
    return -u_xx - u_yy


def output_transform(x, y):
    x_in = x[:, 0:1]
    y_in = x[:, 1:2]

    return x_in * y_in * (1 - x_in) * (1 - y_in) * y


def pde(x, y):
    x_in = x[:, 0:1]
    y_in = x[:, 1:2]
    du_xx = dde.grad.hessian(y, x, i=0, j=0)
    du_yy = dde.grad.hessian(y, x, i=1, j=1)
    return du_xx + du_yy + f(x_in, y_in)


geom = dde.geometry.Rectangle([0, 0], [1, 1])
data = dde.data.PDE(geom, pde, [], num_domain=400)
net = dde.maps.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
net.apply_output_transform(output_transform)

PINNmodel = dde.Model(data, net)

PINNmodel.compile("adam", lr=1.0e-3)
losshistory, train_state = PINNmodel.train(epochs=20000, callbacks=[])

def gen_test_x(num):
    x = np.linspace(0, 1, num)
    y = np.linspace(0, 1, num)
    l = []

    for i in range(len(y)):
        for j in range(len(x)):
            l.append([x[j], y[i]])
    return np.array(l)


def sol(t):
    x = t[:, 0:1]
    y = t[:, 1:2]

    return (16 * x * y * (1 - x) * (1 - y)) ** a


x = gen_test_x(100)
print(
    "L2 relative error of u",
    dde.metrics.l2_relative_error(sol(x), PINNmodel.predict(x)),
)

##########################################
plt.rcParams.update({"font.size": 22})

plt.figure()
X = gen_test_x(100)
y = PINNmodel.predict(x).tolist()

disp = []
prev = X[0][1]
temp = []

for i in range(len(y)):

    if X[i][1] == prev:
        temp.append(y[i][0])
    else:

        prev = X[i][1]

        temp2 = []
        for elem in temp:
            temp2.append((elem))
        disp.append(temp2)
        temp.clear()

        temp.append(y[i][0])
disp.reverse()
plt.figure(figsize=(7, 7))
plt.xlabel("x")
plt.ylabel("y")
plt.title("PINN Prediction")

ax = plt.gca()
im = ax.imshow(disp, extent=(0, 1, 0, 1))
im.set_clim(0, 1)
ax.set_aspect(1)

divider = make_axes_locatable(ax)
width = ax.get_position().width
height = ax.get_position().height
cax = divider.append_axes("right", size="5%", pad=0.2)
plt.colorbar(im, cax=cax)
plt.show()
########################################

plt.figure()
X = x
y = PINNmodel.predict(x).tolist()
y_true = sol(x)

disp = []
prev = X[0][1]
temp = []

for i in range(len(y)):

    if X[i][1] == prev:
        temp.append(abs(y[i][0] - y_true[i][0]))
    else:

        prev = X[i][1]

        temp2 = []
        for elem in temp:
            temp2.append((elem))
        disp.append(temp2)
        temp.clear()

        temp.append(y[i][0])
disp.reverse()
plt.figure(figsize=(7, 7))
plt.xlabel("x")
plt.ylabel("y")
plt.title("PINN Absolute Error of u")

ax = plt.gca()
im = ax.imshow(disp, extent=(0, 1, 0, 1))
ax.set_aspect(1)

divider = make_axes_locatable(ax)
width = ax.get_position().width
height = ax.get_position().height
cax = divider.append_axes("right", size="5%", pad=0.2)
plt.colorbar(im, cax=cax)
plt.show()

# gPINN
a = 10


def f(x, y):
    u_xx = (
        16**a
        * a
        * (a * (1 - 2 * x) ** 2 - 2 * x**2 + 2 * x - 1)
        * ((x - 1) * x * (y - 1) * y) ** a
        / ((x - 1) ** 2 * x**2)
    )
    u_yy = (
        16**a
        * a
        * (a * (1 - 2 * y) ** 2 - 2 * y**2 + 2 * y - 1)
        * ((x - 1) * x * (y - 1) * y) ** a
        / ((y - 1) ** 2 * y**2)
    )
    return -u_xx - u_yy


def f_x(x, y):
    return -(
        (
            16**a
            * a
            * (2 * x - 1)
            * ((x - 1) * x * (y - 1) * y) ** a
            * (
                -2 * a * (2 * a - 1) * (x - 1) ** 2 * x**2 * y
                + (a - 1) * a * (x - 1) ** 2 * x**2
                + (a - 1) * y**4 * (a * (1 - 2 * x) ** 2 - 2 * (x - 1) * x - 2)
                - 2 * (a - 1) * y**3 * (a * (1 - 2 * x) ** 2 - 2 * (x - 1) * x - 2)
                + y**2
                * (
                    (2 * a * (x - 1) * x + a) ** 2
                    + a * (-2 * (x - 1) * x * ((x - 1) * x + 3) - 3)
                    + 2 * (x - 1) * x
                    + 2
                )
            )
        )
        / ((x - 1) ** 3 * x**3 * (y - 1) ** 2 * y**2)
    )


def f_y(x, y):
    return -(
        (
            16**a
            * a
            * (2 * y - 1)
            * ((x - 1) * x * (y - 1) * y) ** a
            * (
                (a - 1) * x**4 * (a * (1 - 2 * y) ** 2 - 2 * (y - 1) * y - 2)
                - 2 * (a - 1) * x**3 * (a * (1 - 2 * y) ** 2 - 2 * (y - 1) * y - 2)
                + x**2
                * (
                    (2 * a * (y - 1) * y + a) ** 2
                    + a * (-2 * (y - 1) * y * ((y - 1) * y + 3) - 3)
                    + 2 * (y - 1) * y
                    + 2
                )
                - 2 * a * (2 * a - 1) * x * (y - 1) ** 2 * y**2
                + (a - 1) * a * (y - 1) ** 2 * y**2
            )
        )
        / ((x - 1) ** 2 * x**2 * (y - 1) ** 3 * y**3)
    )


def output_transform(x, y):
    x_in = x[:, 0:1]
    y_in = x[:, 1:2]

    return x_in * y_in * (1 - x_in) * (1 - y_in) * y


def pde(x, y):
    x_in = x[:, 0:1]
    y_in = x[:, 1:2]
    du_xx = dde.grad.hessian(y, x, i=0, j=0)
    du_yy = dde.grad.hessian(y, x, i=1, j=1)

    du_xxx = dde.grad.jacobian(du_xx, x, j=0)
    du_xxy = dde.grad.jacobian(du_xx, x, j=1)
    du_yyy = dde.grad.jacobian(du_yy, x, j=1)
    du_yyx = dde.grad.jacobian(du_yy, x, j=0)

    return [
        du_xx + du_yy + f(x_in, y_in),
        du_xxx + du_yyx + f_x(x_in, y_in),
        du_xxy + du_yyy + f_y(x_in, y_in),
    ]


geom = dde.geometry.Rectangle([0, 0], [1, 1])
data = dde.data.PDE(geom, pde, [], num_domain=400)
net = dde.maps.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
net.apply_output_transform(output_transform)

gPINNmodel = dde.Model(data, net)

gPINNmodel.compile("adam", lr=1.0e-3, loss_weights=[1, 1e-5, 1e-5])
losshistory, train_state = gPINNmodel.train(epochs=20000, callbacks=[])

def gen_test_x(num):
    x = np.linspace(0, 1, num)
    y = np.linspace(0, 1, num)
    l = []

    for i in range(len(y)):
        for j in range(len(x)):
            l.append([x[j], y[i]])
    return np.array(l)


def sol(t):
    x = t[:, 0:1]
    y = t[:, 1:2]

    return (16 * x * y * (1 - x) * (1 - y)) ** a


x = gen_test_x(100)
print(
    "L2 relative error of u",
    dde.metrics.l2_relative_error(sol(x), gPINNmodel.predict(x)),
)

##########################
plt.rcParams.update({"font.size": 22})

plt.figure()
X = gen_test_x(100)
y = gPINNmodel.predict(x).tolist()

disp = []
prev = X[0][1]
temp = []

for i in range(len(y)):

    if X[i][1] == prev:
        temp.append(y[i][0])
    else:

        prev = X[i][1]

        temp2 = []
        for elem in temp:
            temp2.append((elem))
        disp.append(temp2)
        temp.clear()

        temp.append(y[i][0])
disp.reverse()
plt.figure(figsize=(7, 7))
plt.xlabel("x")
plt.ylabel("y")
plt.title("gPINN Prediction")

ax = plt.gca()
im = ax.imshow(disp, extent=(0, 1, 0, 1))
im.set_clim(0, 1)
ax.set_aspect(1)

divider = make_axes_locatable(ax)
width = ax.get_position().width
height = ax.get_position().height
cax = divider.append_axes("right", size="5%", pad=0.2)

plt.colorbar(im, cax=cax)
plt.show()

########################################

plt.figure()
X = x
y = gPINNmodel.predict(x).tolist()
y_true = sol(x)

disp = []
prev = X[0][1]
temp = []

for i in range(len(y)):

    if X[i][1] == prev:
        temp.append(abs(y[i][0] - y_true[i][0]))
    else:

        prev = X[i][1]

        temp2 = []
        for elem in temp:
            temp2.append((elem))
        disp.append(temp2)
        temp.clear()

        temp.append(y[i][0])
disp.reverse()
plt.figure(figsize=(7, 7))
plt.xlabel("x")
plt.ylabel("y")
plt.title("gPINN Absolute Error of u")

ax = plt.gca()
im = ax.imshow(disp, extent=(0, 1, 0, 1))
im.set_clim(0, 0.004)
ax.set_aspect(1)

divider = make_axes_locatable(ax)
width = ax.get_position().width
height = ax.get_position().height
cax = divider.append_axes("right", size="5%", pad=0.2)
cbar = plt.colorbar(im, cax=cax, ticks=[0, 0.001, 0.002, 0.003, 0.004])
cbar.ax.set_yticklabels([0.000, 0.001, 0.002, 0.003, 0.004])
plt.show()
