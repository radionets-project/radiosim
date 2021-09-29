import numpy as np
import toml
import click


@click.command()
@click.argument("configuration_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--mode",
    type=click.Choice(
        [
            "simulate",
            "overview",
        ],
        case_sensitive=False,
    ),
    default="train",
)
def main(configuration_path, mode):
    config = toml.load(configuration_path)
    sim_conf = read_config(config)

if mode == "simulate":


if mode == "overview":


if __name__ == "__main__":
    main()