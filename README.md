# SymINDy - Symbolic Identification of Nonlinear Dynamics

This library is a generalization of [SINDy](https://github.com/dynamicslab/pysindy), to be used for the reconstruction of dynamical systems with strong nonlinearities, which require the introduction of a combinatorial search in the elementary functions associated with the linear regression part of SINDy.

# About
Let's simultate some well-known dynmical systems and try to reconstruct them with SymINDy.
We will use following dynamical systems:
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
- validation - supplementary module containing the code to reconstruct some dynamical systems are illustrate the performance of the model as well as its comparison with the SINDy algorithm.

Analogously to scikit learn estimators, *SymINDy* main class implements the methods *fit*, *predict*, *score* and *plot_trees*.

## Linear Damped SHO
*To reconstruct the figure below run the script ```reconstruct_linear_damped_sho.py``` from simindy.validation package.*

We simulate Linear Damped SHO on specific time-range, separate the simulated data into the train and test set (train-test ratio 0.7: 0.3). Then we *fit* SymINDy instance on the train set and call the *predict* method on the test set. The predicted data is plotted along with the original data.


## Installing the package
### Option 1
Running the shell script
````bash
bash install symindy
````
### Option 2
1. Create a new python virtual environment
````bash
python -m venv env
````
2. Activate virtual environment
```bash
source env/Scripts/activate
```
3. Downgrade setuptools to version 57.0.0 ([required by DEAP](https://github.com/DEAP/deap/issues/610#issuecomment-1146848490) when using python > 3.7)
```bash
pip install setuptools==57.0.0

````
4. Install the requirements using pip
````bash
pip install -r requirements.txt
````
5. Install the SymINDy package
````bash
pip install -e .
````
### Option 3
Run the package inside of the docker container build from Dockerfile.
````bash
docker build -t SymINDy .
````
Note, you shall have the docker client and daemon installed on your machine.

## Commands
Dear developer, before pushing, remember to run

```commandline
 pre-commit run --all-files
```

## Demo scripts
The demo scripts illustrating the performance of SymINDy on some dynamical systems are located in the validation module (symindy/validation).

## Relevant works

- [Orbital Anomaly Reconstruction Using Deep Symbolic Regression](https://www.researchgate.net/publication/344475621_Orbital_Anomaly_Reconstruction_Using_Deep_Symbolic_Regression)
- [DEAP: Evolutionary Algorithms Made Easy](https://www.jmlr.org/papers/volume13/fortin12a/fortin12a.pdf)
- [PySINDy: A Python package for the sparse identification of nonlinear dynamical systems from data.](https://arxiv.org/abs/2004.08424)
