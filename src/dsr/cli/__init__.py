import click


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
)
def cli():
    """Management CLI for gag"""


@cli.command(name="click_example")
def click_example():
    """Hello world of documentation."""
    print("Hello world, you are using the command line interface of gag!")