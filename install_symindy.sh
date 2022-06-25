#!/bin/bash
# create venv
python -m venv env

# activate venv
source env/Scripts/activate

# downgrade setuptools - requirement of deap for python 3.10
pip install setuptools==57.0.0

# install requirements 
pip install -r requirements.txt

# install SymINDy
pip install -e .
