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

system = nl.myspring# change to different systems e.g. odes.linear_3D
figtitle = 'myspring'

### linear ODEs
time_start=-10
time_end=10
nsamples=1000
time_span=np.linspace(time_start, time_end, nsamples, endpoint=False)
x0=[1,-2.4] # change depending on the dimensionality of the system

dynsys=DynamicalSystem(system, x0=x0)
x, xdot = dynsys.simulate(time_start, time_end, nsamples)

split = lambda x, ratio: (x[:int(ratio*len(x))], x[int(ratio*len(x)):]) if x is not None else (None, None)
ratio = 0.33
x_tr, x_te = split(x, ratio)
xdot_tr, xdot_te = split(xdot, ratio)
time_tr, time_te = split(time_span, ratio)

# istantiate symINDy
symindy = SymINDy(verbose=True, sparsity=None)

# fit symINDy on the training data
symindy.fit(x_tr, xdot_tr, time_tr)

# predict
x_te_pred, xdot_te_pred = symindy.predict(x0, time_te)

# assess predictions via correlation
corr_x = r2_score(x_te, x_te_pred)
corr_xdot = r2_score(xdot_te, xdot_te_pred)
# aggregate the data in a dict
data = {'x_te': x_te, 'x_te_pred':x_te_pred, 'xdot_te': xdot_te, 'xdot_te_pred': xdot_te_pred,
    'x_metric': {"name": "r2", "value": corr_x}, 'xdot_metric': {"name": "r2","value": corr_xdot}}

# plot original and predicted data
fig, ax = plot2d(data, figtitle=figtitle)
#TODO some loops through ~ 10 systems and a figure showing the original and predicted systems.
# Consider seaborn.
plt.show()
print('elapsed')