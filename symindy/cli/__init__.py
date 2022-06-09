import click
from godot.core.tempo import Epoch

import gag.so.HelloWorld_cpp as hw


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
)
def cli():
    """Management CLI for gag"""


@cli.command(name="click_example")
def click_example():
    """Hello world of documentation."""
    print("Hello world, the command line interface of gag works!")


@cli.command(name="godot101")
def godot101():
    """Documentation"""
    epoch = Epoch("2012-12-23T18:23:23.00 TDB")
    Bool = type(epoch) == Epoch
    # print(Bool)
    print("Hello world, the godot is correctly integrated into gag!")


@cli.command(name="HelloWorld_cpp")
def HelloWorld_cpp():
    """pybind11"""
    test = hw.add(1, 3)
    # print(hw.subtract(1, 2))
    # print(hw.matteo_test(5))
    print("Hello world, the cpp scripts of gag can be read from Python!")
