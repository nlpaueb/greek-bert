import click

from .xnli.__main__ import xnli


@click.group()
def cli():
    pass


cli.add_command(xnli)

if __name__ == '__main__':
    cli()
