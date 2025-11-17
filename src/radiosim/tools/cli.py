import rich_click as click

from radiosim import __version__

from .quickstart import main as quickstart
from .simulation_cli import main as sim

click.rich_click.COMMAND_GROUPS = {
    "radiosim": [
        {
            "name": "Simulations",
            "commands": ["simulate"],
        },
        {
            "name": "Setup",
            "commands": ["quickstart"],
        },
    ]
}


@click.group(
    help=f"""
    This is the [dark_turquoise]radiosim[/]
    [cornflower_blue]v{__version__}[/] main CLI tool.
    """
)
def main():
    pass


main.add_command(quickstart, name="quickstart")
main.add_command(sim, name="simulate")

if __name__ == "__main__":
    main()
