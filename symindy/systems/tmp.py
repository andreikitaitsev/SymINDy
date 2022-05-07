import numpy as np
from ode import Lorenz

time=(0,10)
y0=[-8,8,27]
lorenz = Lorenz()
obs = lorenz.simulate(time)
print(obs)