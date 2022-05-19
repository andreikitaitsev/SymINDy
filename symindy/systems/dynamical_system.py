'''Dynamical system base class. 
It is convenient for the usage with pysindy.utils.odes functions.'''
from scipy.integrate import solve_ivp
import numpy as np


class DynamicalSystem:
    def __init__(self, func, x0, solve_ivp_kwargs={'dense_output': True}):

        '''
        Inputs:
            func - a collable object
        '''
        self.func = func
        self.x0 = x0
        self.solve_ivp_kwargs = solve_ivp_kwargs
    
    def simulate(self, t_start, t_end, n_samples):
        t_eval = np.linspace(t_start, t_end, n_samples)
        x = solve_ivp(self.func, t_span=[t_start, t_end], y0=self.x0, t_eval=t_eval,
            **self.solve_ivp_kwargs).y
        xdot = self.func(t_eval, x)
        xdot = np.array(xdot).T
        x = x.T
        return x, xdot