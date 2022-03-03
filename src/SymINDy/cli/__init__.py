import click
from SymINDy.prop import prop


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
)
def cli():
    """Management CLI for SymINDy"""


@cli.command(name="helloworld")
def helloworld():
    """Hello world of documentation."""
    print("Hello world, you are using the command line interface of SymINDy!")


@cli.command(name="propagate")
@click.option(
    "-s",
    "--system",
    required=True,
    help="Dynamical System.",
)
@click.option(
    "-t",
    "--time",
    type=float,
    required=True,
    help="Integration time",
)
# @click.option(
#     "-x0",
#     "--x0",
#     required=True,
#     help="The initial state.",
#     nargs = 2
# )
# @click.option('--pos', , type=float)
def propagate(system, time):
    """propagate a given system and save a series of snapshots in a txt file"""
    prop(system, time)

