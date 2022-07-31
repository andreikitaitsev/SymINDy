'''Reconstruct non-linear dynamical system including sin(x^2) with pySINDy and SymINDy.'''
"""Test the quality of the reconstructuion and prediction of different dynamical systems."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from pathlib import Path
import pysindy as ps
from symindy.symindy import SymINDy
from systems import non_linear_systems as nl
from systems.dynamical_system import DynamicalSystem
from systems.non_linear_systems import lorenz
from validation.utils import plot2d, plot3d, split

## myspring
#    xdot = v
#    vdot = - k x - v c + F sin(x**2)
# create dynamical system and simulate x, xdot
system = nl.myspring
time_start = 0
time_end = 100
nsamples = 200
time_span = np.linspace(time_start, time_end, nsamples, endpoint=False)
x0 = [0.4*100, 1.6*100]  # change depending on the dimensionality of the system
dynsys = DynamicalSystem(system, x0=x0)
x, xdot = dynsys.simulate(time_start, time_end, nsamples)


## SymINDy
# train - test split
ratio = 0.33
x_tr, x_te = split(x, ratio)
xdot_tr, xdot_te = split(xdot, ratio)
time_tr, time_te = split(time_span, ratio)

# istantiate symINDy
symindy = SymINDy(verbose=False, sparsity_coef=1, library_name="generalized")

# fit symINDy on the training data
symindy.fit(x_tr, xdot_tr, time_tr)

# predict
x_te_pred_symindy, xdot_te_pred_symindy = symindy.predict(x_te[0], time_te)

# assess predictions via correlation
corr_x_symindy = r2_score(x_te, x_te_pred_symindy)
corr_xdot_simindy = r2_score(xdot_te, xdot_te_pred_symindy)


## pySINDy
# Note that pysindy does not include out-of-the-box support for the train-test set paradigm,
# therefore, we train it on the data used as a test set for symindy
sindy = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=2))
sindy.fit(x_te, time_te, xdot_te, quiet=True)

# simulate x
x_te_pred_sindy = sindy.simulate(x_te[0], t=time_te)
# simulate xdot
xdot_te_pred_sindy = sindy.predict(x_te_pred_sindy)

# assess predictions via correlation
corr_x_sindy = r2_score(x_te, x_te_pred_sindy)
corr_xdot_sindy = r2_score(xdot_te, xdot_te_pred_sindy)

# plot original systems and predictions os symindy and pysindy
fig, axs = plt.subplots(2)
axs[0].plot(x_te, color='b')
axs[0].plot(x_te_pred_symindy, color='r')
axs[0].plot(x_te_pred_sindy, color='g')
axs[0].legend(['original','symindy','sindy'])

axs[1].plot(xdot_te, color='b')
axs[1].plot(xdot_te_pred_symindy, color='r')
axs[1].plot(xdot_te_pred_sindy, color='g')
axs[1].legend(['original','symindy','sindy'])
plt.show()
