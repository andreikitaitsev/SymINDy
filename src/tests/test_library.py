from unittest import TestCase
from symindy.library import Library
import deap


class TestLibrary(TestCase):

    def setUp(self):
        self.library = Library(0, 2)
        self.library.create_pset()

    def test_create_pset(self):
        self.assertTrue(isinstance(self.library.pset, deap.gp.PrimitiveSetTyped))
    
    def test_polynomial_library(self):
        self.library.polynomial_library()
        self.assertTrue(self.library.pset.context["mul"].__name__ == "multiply")
        self.assertTrue(self.library.pset.context["add"].__name__ == "add")
    
    def test_fourier_library(self):
        self.library.fourier_library()
        self.assertTrue(self.library.pset.context["sin"].__name__ == "multiply")
        self.assertTrue(self.library.pset.context["cos"].__name__ == "add")

    def test_fourier_library(self):
        self.library.fourier_library()
        self.assertTrue(self.library.pset.context["sin"].__name__ == "sin")
        self.assertTrue(self.library.pset.context["cos"].__name__ == "cos")
    
    def test_generalized_library(self):
        self.library.generalized_library()
        self.assertTrue(all([funcs in self.library.pset.context.keys() for 
            funcs in ["sin", "cos", "add", "mul"] ]))

    def test_call(self):
        pset = self.library()
        self.assertTrue(isinstance(pset, deap.gp.PrimitiveSetTyped))
        