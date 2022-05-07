class BaseODE:
    '''Base class for the simulations of Ordinary Differential equations.'''
    def __init__(self, func, func_params, y0, kwargs={}):
        '''
        Inputs:
            func - ODE function to be integrated via scipy's solve_ivp
            y0 - inital solution
            kwargs for solve_ivp 
        '''
        self.func = func(**func_params)
        self.kwargs = kwargs
        self.y0 = y0
    
    def simulate(self, timespan):
        '''Simulates the time series for a given timespan'''
        self.simulated_timeseries = solve_ivp(self.func, t_span = timespan,
             y0 = self.y0, **self.kwargs)


class Lorenz(BaseODE):
    def __init__(self, func, y0, kwargs):
        super().__init__(self.func, y0, kwargs)
    
    @staticmethod
    def func(z, t):
        '''Lorenz attractor.'''
        return [10*(z[1] - z[0]),
            z[0]*(28 - z[2]) - z[1],
            z[0]*z[1] - 8/3*z[2]]

