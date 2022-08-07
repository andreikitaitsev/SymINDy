from unittest import TestCase
from systems.dynamical_system import DynamicalSystem
from systems.non_linear_systems import myspring, lorenz
import deap
import numpy as np
from pathlib import Path
import pysindy.utils.odes as odes


class TestLibrary(TestCase):

    def setUp(self):
        inp_dir = Path(__file__).parents[1].joinpath("test_resources")
        with open(inp_dir.joinpath('x_train.test_txt'), 'r') as fl:
            self.x = np.loadtxt(fl)
        with open(inp_dir.joinpath('x_dot_train.test_txt'), 'r') as fl:
            self.xdot = np.loadtxt(fl)
        with open(inp_dir.joinpath('time_train.test_txt'), 'r') as fl:
            self.time = np.loadtxt(fl)

    def test_simulate(self):
        self.func = odes.linear_damped_SHO
        self.system = DynamicalSystem(self.func, self.x[0])
        t_start = self.time[0]
        t_end = self.time[-1]
        n_samples = len(self.time)
        self.x_sim, self.xdot_sim = self.system.simulate(t_start, t_end, n_samples)
        self.assertEqual(self.x.shape, self.x_sim.shape)
        self.assertEqual(self.xdot.shape, self.xdot_sim.shape)

    def test_myspring(self):
        x0 = [0.4, 1.6]
        func = myspring
        system = DynamicalSystem(func, x0)
        t_start = self.time[0]
        t_end = self.time[-1]
        n_samples = len(self.time)
        x_sim, xdot_sim = system.simulate(t_start, t_end, n_samples)
        self.assertEqual(x_sim.shape[1], 2) # 2 dimensions
        self.assertEqual(xdot_sim.shape[1], 2)

    def test_lorenz(self):
        x0 = [-8, 8, 27]
        func = lorenz
        system = DynamicalSystem(func, x0)
        t_start = self.time[0]
        t_end = self.time[-1]
        n_samples = len(self.time)
        x_sim, xdot_sim = system.simulate(t_start, t_end, n_samples)
        self.assertEqual(x_sim.shape[1], 3) # 3 dimensions
        self.assertEqual(xdot_sim.shape[1], 3)