from setuptools import find_packages, setup

setup(name='symindy', 
    version='0.1.0', 
    description="Symbolic Identification of Nonlinear Dynamics",
    author="Andrei Kitaitsev, Matteo Manzi",
    author_email="andre.kit17@gmail.com, matteomanzi09@gmail.com ",
    keywords="pysindy, symbolic regression, symindy, dynamical system",
    package_dir={"": "src"},
    url = "https://github.com/andreikitaitsev/SymINDy",
    packages=find_packages(where="src"),
    python_requires="<=3.9.13",
    license = "MIT"
    )
