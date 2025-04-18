import click
import numpy as np
import toml

from radiosim.simulations import simulate_sky_distributions
from radiosim.utils import check_outpath, read_config


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
    default="simulate",
)
def main(configuration_path, mode) -> None:
    config = toml.load(configuration_path)
    conf = read_config(config)
    print(conf, "\n")

    if conf["seed"] != "none":
        click.echo(f"Using numpy random seed {conf['seed']}. \n")
        np.random.seed(conf["seed"])

    outpath = config["paths"]["outpath"]
    sim_sources = check_outpath(
        outpath,
        quiet=config["general"]["quiet"],
    )

    if mode == "simulate":
        if sim_sources:
            simulate_sky_distributions(conf)


if __name__ == "__main__":
    main()
