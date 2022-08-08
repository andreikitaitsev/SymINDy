# SymINDy - Symbolic Identification of Nonlinear Dynamics

This library is a generalization of [SINDy](https://github.com/dynamicslab/pysindy), to be used for the reconstruction of dynamical systems with strong nonlinearities, which require the introduction of a combinatorial search in the elementary functions associated with the linear regression part of SINDy.

# About

Let's simulate some well-known dynamical systems and try to reconstruct them with SymINDy.
We will use the following dynamical systems:

- [Linear Damped SHO](https://en.wikipedia.org/wiki/Duffing_equation)
- [Cubic Damped SHO](https://en.wikipedia.org/wiki/Duffing_equation)
- [Lorenz attractor](https://en.wikipedia.org/wiki/Lorenz_system)
- myspring (author defined perturbing oscillator)
  The first three dynamical systems [were reconstructed by SINDy](https://pysindy.readthedocs.io/en/latest/examples/3_original_paper.html), however, the last system serves as a good example to illustrate the limitations of SINDy related to the linearity of the reconstruction algorithm.
  We will go through each of the systems step by step.

# Project architecture

We follow [scikit learn conventions](https://scikit-learn.org/stable/developers/develop.html) and best practices for the [development of python packages](https://packaging.python.org/en/latest/tutorials/packaging-projects/).
The source code is located in the src directory. It consists of several modules:

- symindy - main module containing SymINDy class
- systems - supplementary module containing classes to conveniently reconstruct dynamical systems
- validation - supplementary module containing the code to reconstruct some dynamical systems and illustrate the performance of the model as well as its comparison with the SINDy algorithm.

Analogously to scikit learn estimators, _SymINDy_ main class implements the methods _fit_, _predict_, _score_ and _plot_trees_.

## Linear Damped SHO

_To reconstruct the figure below run the script `reconstruct_linear_damped_sho.py` from simindy.validation package._

We simulate Linear Damped SHO on a specific time range, separate the simulated data into the train and test set (train-test ratio 0.7: 0.3). Then we _fit_ SymINDy instance on the train set and call the _predict_ method on the test set. The predicted data is plotted along with the original data.
Running the script prints the reconstructed equation of the dynamical system to std output.

```
Estimated library functions: f0: x0
Estimated library functions: f1: x1
Estimated library functions: f2: mul(x0, x0)
Estimated library functions: f3: mul(x0, sin(x0))
Estimated library functions: f4: cos(x0)
Estimated dynamical system:

(y0)' = -0.100 f0(x0,x1) + 2.000 f1(x0,x1)
(y1)' = -2.000 f0(x0,x1) + -0.100 f1(x0,x1)
```

The original equation:

```
y0 = -0.1 * x0 + 2 * x1
y1 = -2 * x0 - 0.1 * x1
```

<div style="margin:2%";>  
    <img src="src\validation\figures\linear_damped_SHO.svg"; alt="linear_damped_SHO"; width=50%;/>
</div>

We correctly reconstruct the numeric equation and predict the test data.
The coefficient of determination between original and reconstructed data is 1.0.

## Cubic Damped SHO

_To reconstruct the figure below run the script `reconstruct_cubic_damped_sho.py` from simindy.validation package._

Same as before, we simulate Cubic Damped SHO on a specific time range, separate the simulated data into the train and test set (train-test ratio 0.7: 0.3). Then we _fit_ SymINDy instance on the train set and call the _predict_ method on the test set. The predicted data is plotted along with the original data.
Running the script prints the reconstructed equation of the dynamical system to std output.

```
Estimated library functions: f0: mul(mul(x1, x1), x1)
Estimated library functions: f1: add(x0, x1)
Estimated library functions: f2: add(x1, add(x0, x0))
Estimated library functions: f3: mul(x0, mul(x0, x0))
Estimated library functions: f4: mul(x0, add(x0, x1))
Estimated dynamical system:

(y0)' = 2.000 f0(x0,x1) + -0.100 f3(x0,x1)
(y1)' = -0.100 f0(x0,x1) + -2.000 f3(x0,x1)
```

The original equation:

```
y0 = -0.1 * x0**3 + 2 * x1**3

y1 = -2 * x0**3 - 0.1 * x1**3
```

<div style="margin:2%";>  
    <img src="src\validation\figures\cubic_damped_SHO.svg"; alt="cubic_damped_SHO"; width=50%;/>
</div>

Again, we correctly reconstruct the numeric equation and predict the test data.
The coefficient of determination between original and reconstructed data is 1.0.

## Lorenz attractor

_To reconstruct the figure below run the script `reconstruct_lorenz.py` from symindy.validation package._

Same as before, we simulate Lorenz Attractor on specific a time range, separate the simulated data into the train and test set (train-test ratio 0.7: 0.3). Then we _fit_ SymINDy instance on the train set and call the _predict_ method on the test set. The predicted data is plotted along with the original data.
Running the script prints the reconstructed equation of the dynamical system to std output.

```
Estimated library functions: f0: mul(x1, x0)
Estimated library functions: f1: x0
Estimated library functions: f2: x1
Estimated library functions: f3: x2
Estimated library functions: f4: mul(x0, x2)
Estimated dynamical system:

(y0)' = -10.000 f1(x0,x1,x2) + 10.000 f2(x0,x1,x2)
(y1)' = 28.000 f1(x0,x1,x2) + -1.000 f2(x0,x1,x2) + -1.000 f4(x0,x1,x2)
(y2)' = 1.000 f0(x0,x1,x2) + -2.667 f3(x0,x1,x2)
```

The original equation:

```
(y0)' =  10*(z1 - z0),
(y1)' =  z0*(28 - z2) - z1,
(y2)' =  z0*z1 - 8/3*z2
```

<div style="margin:2%";>  
    <img src="src\validation\figures\lorenz.svg"; alt="Lorenz"; width=50%;/>
</div>

Again, we correctly reconstruct the numeric equation and predict the test data.
The coefficient of determination between original and reconstructed data is 1.0.

Thus, we accurately reconstruct dynamical linear and non-linear dynamical systems using symbolic regression to look for basis (library) functions. However, the main advantage of SymINDy is revealed with the reconstruction of highly non-linear systems.

## Myspring (nonlinearly perturbed oscillator)

_To reconstruct the figure below run the script `reconstruct_myspring.py` from simindy.validation package._
Again, the fitting procedure is same as above.
However, this time we focus on the original system first.

    (x0)' = x1
    (x1)' = - -4.518 x0 - 0.372 x1 + 9.123*sin(x0**2)

As we can see, the argument passed to sine is also non-linear. This makes it impossible for SINDy to reconstruct the system.
But SymINDy can easyliy do it. Let's run the code `reconstruct_myspring.py` and see how it performs!

Reconstructed equation:

```
Estimated library functions: f0: x1
Estimated library functions: f1: x0
Estimated library functions: f2: cos(x1)
Estimated library functions: f3: sin(mul(x0, x0))
Estimated library functions: f4: sin(mul(x1, x0))
Estimated dynamical system:

(x0)' = 1.000 f0(x0,x1)
(x1)' = -0.372 f0(x0,x1) + -4.518 f1(x0,x1) + 9.123 f3(x0,x1)
```

Bingo, we did it! Let's see the graph.

<div style="margin:2%";>  
    <img src="src\validation\figures\myspring.svg"; alt="myspring"; width=50%;/>
</div>

We have correctly reconstructed non-linearly perturbed oscillator.

But let us not take anything for granted: maybe SINDy would do the same job.
Let's reconstruct _mysping_ with both SINDy and SymINDy and compare the results.

## SymINDy and SINDy

_To reconstruct the figure below run the script `SINDy_vs_SymINDy.py` from simindy.validation package._

<div style="margin:2%";>  
    <img src="src\validation\figures\symindy_vs_sindy.svg"; alt="myspring"; width=50%;/>
</div>

Well, probably our statement above still holds: SINDy cannot estimate non-linearly perturbed oscillator, while SymINDy, as we have seen above, accurately recovers the underlying equation.

## Summary

SymINDy is a new algorithm for the reconstruction of non-linear dynamics. It uses symbolic regression and SINDy algorithm to recover the systems of equations from time-series observations.
It is free from the linearity assumption and thus is able to reconstruct systems unreachable for SINDy.

SymINDy can be applied to multiple theoretical and applied problems from blood dynamics to financial forecasting.

<br>
<br>

## Installing the package

#### Option 1

Running the shell script

```bash
bash install symindy
```

#### Option 2

1. Create a new python virtual environment

```bash
python -m venv env
```

2. Activate virtual environment

```bash
source env/Scripts/activate
```

3. Downgrade setuptools to version 57.0.0 ([required by DEAP](https://github.com/DEAP/deap/issues/610#issuecomment-1146848490) when using python > 3.7)

```bash
pip install setuptools==57.0.0

```

4. Install the requirements using pip

```bash
pip install -r requirements.txt
```

5. Install the SymINDy package

```bash
pip install -e .
```

#### Option 3

Run the package inside of the docker container built from Dockerfile.

```bash
docker build -t SymINDy .
```

Note, you shall have the docker client and daemon installed on your machine.

## Commands

Dear developer, before pushing, remember to run

```commandline
 pre-commit run --all-files
```

## Notes

- _plot_trees_ method requires requires pygraphviz to be installed which is often installed separately. This may not be straightforawrd for windows.
  https://stackoverflow.com/questions/40809758/howto-install-pygraphviz-on-windows-10-64bit
- _is_time_dependent_ attribute is in a test mode and does not generally work. This parameter is only present in _is_time_dependent_ branch and is missing in the _main_ branch

## Relevant works

- [Orbital Anomaly Reconstruction Using Deep Symbolic Regression](https://www.researchgate.net/publication/344475621_Orbital_Anomaly_Reconstruction_Using_Deep_Symbolic_Regression)
- [DEAP: Evolutionary Algorithms Made Easy](https://www.jmlr.org/papers/volume13/fortin12a/fortin12a.pdf)
- [PySINDy: A Python package for the sparse identification of nonlinear dynamical systems from data.](https://arxiv.org/abs/2004.08424)
