import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def prop(system, time):
    if system == "myspringpip":
        x0 = np.array([0.4, 1.6])
        k = 4.518
        c = 0.376
        F0 = 8.865

        # xdot = v
        # vdot = - k x - v c + A sin(x**2)
        def dxdt(t, x):
            return [x[1], -k * x[0] - c * x[1] + F0 * np.sin(x[0] ** 2)]

        t_eval = np.linspace(0, time, 200)
        obs = solve_ivp(dxdt, y0=x0, t_span=[0, time], t_eval=t_eval).y

        fig, axs = plt.subplots(len(x0), 1)
        axs[0].plot(t_eval, obs[0, :])
        axs[1].plot(t_eval, obs[1, :])
        plt.savefig("myspring.png")
        # SymINDy (pysindy) requires x to be of shape (timesteps, systems (equations))
        obs = obs.T
        np.savetxt("myspring.txt", obs)
