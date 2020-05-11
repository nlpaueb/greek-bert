import click

from .xnli.__main__ import xnli
from .ud.__main__ import ud


@click.group()
def cli():
    pass


cli.add_command(xnli)
cli.add_command(ud)

if __name__ == '__main__':
    cli()
