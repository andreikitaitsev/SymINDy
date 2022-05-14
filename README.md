# SIND_and_symbolic_regression

## Dependencies

TODO: move this inside the setup.py, and remove the need to run it.

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

## Notes

This software is based on the one used [here](https://www.researchgate.net/publication/344475621_Orbital_Anomaly_Reconstruction_Using_Deep_Symbolic_Regression?_sg%5B0%5D=0wBU1i2FuNFv7GrI5tBwlNlsXa1lTJSK3rr_wah32-TZA0DthKdFWvdgDnhpa4j9zw4oxvvYCXRlm-dut4Ex33DScJfQ7oLG-5lmh-vk.qBVh3aGjqIRH7w-Nv8p7oDqT05hMsVnx7MgIomyCJsV_xdfT0YrIb-Tjm2I3-AnyS49FuI-t7qR0m5asIPW71g).

It combines [deap](https://deap.readthedocs.io/en/master/) with [PySINDy](https://pysindy.readthedocs.io/en/latest/).

SINDy is nice because it allows one to leverage linear ideas, while deap takes care of nonlinearities (to some extent).
See also works by Brenden (LLNL) and Max Tegmark (MIT). Deep Symbolic Regression, AI Feynman.
See also work by Facebook AI.