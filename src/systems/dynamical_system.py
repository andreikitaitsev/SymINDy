'''Dynamical system base class.
It is convenient for the usage with pysindy.utils.odes functions.'''
import numpy as np
from scipy.integrate import solve_ivp


class DynamicalSystem:
    def __init__(self, func, x0, solve_ivp_kwargs={'dense_output': True}):
        '''
        Dynamical system class makes it convenient to simulate time observations
        from dynamical system specified as function.
        Parameters:
            func - a collable object defining dynamical system
            x0 - float or np array, first observation of the dynamical system
            solve_ivp_kwargs - dict, arguments to scipy solve_ivp
        '''
        self.func = func
        self.x0 = x0
        self.solve_ivp_kwargs = solve_ivp_kwargs

    def simulate(self, t_start, t_end, n_samples):
        '''
        Simulate time-series of observations.
        Parameters:
            t_start - int, start of observation time
            t_end - int, end of observation time
            n_samples - int, number of samples if observations
        Returns:
            x, xdor - numpy array, simulated observations of dynamical system 
                and their derivatives.
        '''
        t_eval = np.linspace(t_start, t_end, n_samples, endpoint=False)
        x = solve_ivp(self.func, t_span=[t_start, t_end], y0=self.x0, t_eval=t_eval,
            **self.solve_ivp_kwargs).y
        xdot = self.func(t_eval, x)
        xdot = np.array(xdot).T
        x = x.T
        return x, xdot
