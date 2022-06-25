# SymINDy - Symbolic Identification of Nonlinear Dynamics

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
