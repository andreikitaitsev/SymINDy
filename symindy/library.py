import numpy as np
from deap import gp


class library:
    def __init__(self, nc, dimensions, is_time_dependent, library_name="generalized"):
        self.nc = nc
        self.dimensions = dimensions
        self.library_name = library_name
        self.is_time_dependent = is_time_dependent

    def create_pset(self):
        size_input = self.dimensions + self.nc
        # TODO let the dimensionality be a function of an input file
        if self.is_time_dependent:
            size_input += 1
        intypes = [float for i in range(size_input)]
        # 1)name, 2)type of each input, 3)type of the output
        pset = gp.PrimitiveSetTyped("MAIN", intypes, float)  
        self.pset = pset

    def polynomial_library(self):
        self.pset.addPrimitive(np.multiply, [float, float], float, name="mul")
        self.pset.addPrimitive(np.add, [float, float], float, name="add")

    def fourier_library(self):
        self.pset.addPrimitive(np.sin, [float], float, name="sin")
        self.pset.addPrimitive(np.cos, [float], float, name="cos")

    def generalized_library(self):
        self.polynomial_library()
        self.fourier_library()
        # call all the libraries

    def __call__(self):
        self.create_pset()
        if self.library_name == "polynomial":
            self.polynomial_library()
        elif self.library_name == "fourier":
            self.fourier_library()
        elif self.library_name == "generalized":
            self.generalized_library()
        return self.pset