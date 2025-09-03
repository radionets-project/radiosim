import click
import numpy as np
import toml
from rich.pretty import pretty_repr

from radiosim.logging import setup_logger
from radiosim.simulations import simulate_sky_distributions
from radiosim.utils import check_outpath, read_config

LOGGER = setup_logger(namespace=__name__, tracebacks_suppress=[click])


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

    LOGGER.info("Starting simulation of radio sky distributions:")
    LOGGER.info(pretty_repr(conf))

    if conf["seed"] != "none":
        LOGGER.info(f"Using numpy random seed {conf['seed']}.")
        np.random.seed(conf["seed"])

    outpath = config["paths"]["outpath"]
    sim_sources = check_outpath(
        outpath,
        quiet=config["general"]["quiet"],
    )

    if mode == "simulate" and sim_sources:
        simulate_sky_distributions(conf)


if __name__ == "__main__":
    main()
