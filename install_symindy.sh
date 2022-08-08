python3 -m venv venv

source venv/bin/activate

# downgrade setuptools - requirement of deap for python 3.10
pip install setuptools==57.0.0

# install requirements
pip install -r requirements.txt

# install SymINDy
pip install .
