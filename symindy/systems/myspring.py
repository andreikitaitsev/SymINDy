from scipy.integrate import solve_ivp
import numpy as np

class myspring:
    '''
    xdot = v
    vdot = - k x - v c + A sin(x**2)
    '''
    def __init__(self,
        x0=np.array([0.4, 1.6]),
        k=4.518,
        c=0.372,
        F0=9.123,
        time=10,
        nsamples=200
        ):
        self.x0=x0
        self.k=k
        self.c=c
        self.F0=F0
        self.time=time
        self.nsamples=nsamples

    def simulate(self):
        '''Reconstruct differential equation'''
        def dxdt(t, x):
            return [x[1], -self.k * x[0] - self.c * x[1] + self.F0 * np.sin(x[0] ** 2)]
        t_eval = np.linspace(0, self.time, self.nsamples)
        x = solve_ivp(dxdt, y0=self.x0, t_span=[0, self.time], t_eval=t_eval).y
        x = x.T
        xdot = None
        return x, xdot, t_eval