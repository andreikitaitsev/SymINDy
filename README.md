# SIND_and_symbolic_regression

## Requirements for publication

- [ ] Use our library to reconstruct the Lorenz Attractor as in the original SINDy paper.
- [ ] Reconstruct a 1D but "complex ode" (e.g., xdot = - k x + A sin(omega sqrt(x))) using the real number mutation in SymbolicRegression.jl
- [ ] Reconstruct the dynamics in [here](https://www.researchgate.net/publication/344475621_Orbital_Anomaly_Reconstruction_Using_Deep_Symbolic_Regression?_sg%5B0%5D=0wBU1i2FuNFv7GrI5tBwlNlsXa1lTJSK3rr_wah32-TZA0DthKdFWvdgDnhpa4j9zw4oxvvYCXRlm-dut4Ex33DScJfQ7oLG-5lmh-vk.qBVh3aGjqIRH7w-Nv8p7oDqT05hMsVnx7MgIomyCJsV_xdfT0YrIb-Tjm2I3-AnyS49FuI-t7qR0m5asIPW71g).
- [ ] Reconstruct the two-body problem using the norm operation acting on vectors (the inputs are still scalar, so there are primives building vectors from components)
- [ ] Reconstruct a system in which cross and dot product appear.
- [ ] Reconstruct a non-autonomous system using a flag as an input "try nonautonomous" (e.g., \ddot{x} = - k x + sin (sin(\omega t)))
- [ ] Create front-end application
- [ ] Sphinx Documentation

## Dependencies

In order to install dependencies, run the following:

```commandline
pip install pip-tools
```

and then

```commandline
pip install -r requirements/dev.txt
```

## Commands
Before pushing, remember to run
```commandline
 pre-commit run --all-files
```

```commandline
SymINDy propagate -s myspring -t 10
```

## High-level goals

- Passing the Build CI
- Passing the documentation (sphinx recommended, but open to alternatives)
- Have command-line interface (click recommended, but open to alternatives)
- Publish paper on JOSS
- Publish video tutorial on Youtube
- Ask to be mentioned in the related packages [here](https://github.com/dynamicslab/pysindy)

## Notes

It combines [deap](https://deap.readthedocs.io/en/master/) with [PySINDy](https://pysindy.readthedocs.io/en/latest/).

SINDy is nice because it allows one to leverage linear ideas, while deap takes care of nonlinearities (to some extent).
See also works by Brenden (LLNL) and Max Tegmark (MIT). Deep Symbolic Regression, AI Feynman.
See also work by Facebook AI.

Can we also do something with divergence? Gradient? Do we need measurements in space as well to compute the partials with finite difference? In this way we can have a bigger library.
