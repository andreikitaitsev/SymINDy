import click


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
)
def cli():
    """Management CLI for SymINDy"""


@cli.command(name="helloworld")
def helloworld():
    """Hello world of documentation."""
    print("Hello world, you are using the command line interface of SymINDy!")