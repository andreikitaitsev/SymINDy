"""Test the quality of the reconstructuion and prediction of different dynamical systems."""
from tkinter import N
import matplotlib.pyplot as plt
import numpy as np
import pysindy.utils.odes as odes
from sklearn.metrics import r2_score
from pathlib import Path

from symindy.symindy import SymINDy
from systems import non_linear_systems as nl
from systems.dynamical_system import DynamicalSystem
from systems.non_linear_systems import lorenz
from validation.utils import plot2d, plot3d, split


## linear_damped_SHO
# create dynamical system and simulate x, xdot
system = odes.linear_damped_SHO
time_start = 0
time_end = 10
nsamples = 200
time_span = np.linspace(time_start, time_end, nsamples, endpoint=False)
x0 = [2, 0]  # change depending on the dimensionality of the system
dynsys = DynamicalSystem(system, x0=x0)
x, xdot = dynsys.simulate(time_start, time_end, nsamples)

# train - test split
ratio = 0.33
x_tr, x_te = split(x, ratio)
xdot_tr, xdot_te = split(xdot, ratio)
time_tr, time_te = split(time_span, ratio)

# istantiate symINDy
symindy = SymINDy(sparsity_coef=0.05, library_name="generalized")

# fit symINDy on the training data
symindy.fit(x_tr, xdot_tr, time_tr)

# predict
x_te_pred, xdot_te_pred = symindy.predict(x_te[0], time_te)

# assess predictions via correlation
corr_x = r2_score(x_te, x_te_pred)
corr_xdot = r2_score(xdot_te, xdot_te_pred)

# aggregate the data in a dict
data = {
    "x_te": x_te,
    "x_te_pred": x_te_pred,
    "xdot_te": xdot_te,
    "xdot_te_pred": xdot_te_pred,
    "x_metric": {"name": "r2", "value": corr_x},
    "xdot_metric": {"name": "r2", "value": corr_xdot},
    "time": time_te
}

# plot original and predicted data
fig, ax = plot2d(data, figtitle="Linear Damped SHO")
out_dir=Path(__file__).parents[0].joinpath('figures')
if not out_dir.is_dir():
    out_dir.mkdir()
fig.savefig(out_dir.joinpath('linear_damped_SHO.svg'), dpi=300)
plt.show()
