'''Reconstruct non-linear dynamical system including sin(x^2) with pySINDy and SymINDy.'''
"""Test the quality of the reconstructuion and prediction of different dynamical systems."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
from sklearn.metrics import r2_score

from symindy.symindy import SymINDy
from systems import non_linear_systems as nl
from systems.dynamical_system import DynamicalSystem
from validation.utils import plot_compare_sindy_simindy, split

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

data = {
    "x_te": x_te,
    "x_te_pred_symindy": x_te_pred_symindy,
    "x_te_pred_sindy": x_te_pred_sindy,
    "xdot_te": xdot_te,
    "xdot_te_pred_symindy": xdot_te_pred_symindy,
    "xdot_te_pred_sindy": xdot_te_pred_sindy
    }

# plot
fig, axs = plot_compare_sindy_simindy(data, figtitle="SymINDy vs SINDy")

# save the figure
out_dir=Path(__file__).parents[0].joinpath('figures')
if not out_dir.is_dir():
    out_dir.mkdir()
fig.savefig(out_dir.joinpath('symindy_vs_sindy.svg'), dpi=300)
plt.show()
