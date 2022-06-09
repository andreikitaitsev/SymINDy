from setuptools import find_packages, setup

AUTHOR = "Andrei Kitaitsev, Matteo Manzi"
VERSION = "0.1"

setup(
    name="symindy",
    version=VERSION,
    author=AUTHOR,
    packages=find_packages("symindy"),
    package_dir={"": "symindy"},
    include_package_data=True,
    python_requires=">= 3.8",
    entry_points={"console_scripts": ["symindy=symindy.__main__:cli"]},
)