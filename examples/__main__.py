import click

from .xnli.__main__ import xnli
from .ud.__main__ import ud
from .ner.__main__ import ner


@click.group()
def cli():
    pass


cli.add_command(xnli)
cli.add_command(ud)
cli.add_command(ner)

if __name__ == '__main__':
    cli()
