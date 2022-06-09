'''Test the quality of the reconstructuion and prediction of different dynamical systems.'''
from symindy.systems.non_linear_systems import lorenz
from symindy.systems.dynamical_system import DynamicalSystem
from symindy.symindy.symindy import SymINDy
import pysindy.utils.odes as odes
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from symindy.validation.utils import plot2d, plot3d
from symindy.systems import non_linear_systems as nl

system = nl.lorenz # change to different systems e.g. odes.linear_3D  #nl.lorenz #

### linear ODEs
time_start=0
time_end=100
nsamples=10000
time_span=np.linspace(time_start, time_end, nsamples, endpoint=False)
x0=[-8, 8, 27] # change depending on the dimensionality of the system

dynsys = DynamicalSystem(system, x0=x0)
x, xdot = dynsys.simulate(time_start, time_end, nsamples)

split = lambda x, ratio: (x[:int(ratio*len(x))], x[int(ratio*len(x)):]) if x is not None else (None, None)
ratio = 0.33
x_tr, x_te = split(x, ratio)
xdot_tr, xdot_te = split(xdot, ratio)
time_tr, time_te = split(time_span, ratio)

# istantiate symINDy
symindy = SymINDy(verbose=False, sparsity="n_zero_nodes")

# fit symINDy on the training data
symindy.fit(x_tr, xdot_tr, time_tr)

# predict
x_te_pred, xdot_te_pred = symindy.predict(x_te[0], time_te)

# assess predictions via correlation
corr_x = r2_score(x_te, x_te_pred)
corr_xdot = r2_score(xdot_te, xdot_te_pred)

# plot x train
fig = plt.figure(figsize=(15, 4))
plt.title("lorenz")

ax = fig.add_subplot(121, projection="3d")
ax.plot(
    x_te[:, 0],
    x_te[:, 1],
    x_te[:, 2], color="b"
)

# plot x test
ax = fig.add_subplot(122, projection="3d")
ax.plot(
   x_te_pred[:, 0],
   x_te_pred[:, 1],
   x_te_pred[:, 2], color='r'
)

ax.legend(["x te", "x te pred"])
ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")

plt.show()
