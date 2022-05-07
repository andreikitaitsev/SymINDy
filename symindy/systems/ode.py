'''ODE systems'''
from scipy.integrate import solve_ivp
import numpy as np


class Lorenz:
    '''
    [sigma*(z[1] - z[0]),
    z[0]*(rho - z[2]) - z[1],
    z[0]*z[1] - beta*z[2]]
    '''
    def __init__(self, y0=[3,-1,-12.3], sigma=10, beta=3, rho=2, 
        solve_ivp_kwargs={'dense_output': True} ):
        self.y0 = y0
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.solve_ivp_kwargs = solve_ivp_kwargs

    def simulate(self, timespan):
        '''Simulate lorenz attractor timeseries.
        Inputs:
            Timespan - (start, end)
        '''
        def lorenz(t, X, sigma, beta, rho):
            """The Lorenz equations."""
            u, v, w = X
            up = -sigma*(u - v)
            vp = rho*u - v - u*w
            wp = -beta*w + u*v
            return up, vp, wp
        solution = solve_ivp(lorenz, timespan, self.y0, args=(self.sigma, self.beta, self.rho),
            **self.solve_ivp_kwargs)
        return solution.y