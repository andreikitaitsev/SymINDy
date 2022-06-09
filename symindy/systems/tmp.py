import numpy as np

from symindy.symindy.symindy import SymINDy_class
from symindy.systems.dynamical_system import DynamicalSystem
from symindy.systems.non_linear_systems import lorenz

time=(0,10)
y0=[-8,8,27]
x0=[1,-2,2.4]
dynsys=DynamicalSystem(lorenz, x0=x0)
x, xdot = dynsys.simulate(0, 100, 200)
split = lambda x, ratio: (x[:int(ratio*len(x))], x[int(ratio*len(x)):]) if x is not None else (None, None)
ratio = 0.33
x_tr, x_te = split(x, ratio)
xdot_tr, xdot_te = split(xdot, ratio)
time_tr, time_te = split(time, ratio)

symindy = SymINDy_class(verbose=True)

symindy.fit(x_tr, xdot_tr)
x_te_pred, xdot_te_pred = symindy.predict(x0, time_te)
