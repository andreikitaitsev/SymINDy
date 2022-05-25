'''Noin-linear dynamical systems base class.'''
import numpy as np

def myspring(t, x, k=-4.518, c=0.372, F0=9.123):
    '''
    Example nonlinear dynamical system.
    xdot = v
    vdot = - k x - v c + F sin(x**2)
    '''
    return [x[1], k * x[0] - c * x[1] + F0 * np.sin(x[0] ** 2)]

def lorenz(t, x, sigma=10, beta=3, rho=2):
    '''
    Simulate lorenz attractor timeseries.
    Lorenz equations.
    [sigma*(z[1] - z[0]),
    z[0]*(rho - z[2]) - z[1],
    z[0]*z[1] - beta*z[2]]
    '''
    u, v, w = x
    up = -sigma*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    return up, vp, wp