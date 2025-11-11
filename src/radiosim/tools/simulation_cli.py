import numpy as np
import rich_click as click
from rich.pretty import pretty_repr

from radiosim.io import Config
from radiosim.logging import setup_logger
from radiosim.simulations import simulate_ppdisks, simulate_sky_distributions
from radiosim.utils import check_outpath

LOGGER = setup_logger(namespace=__name__, tracebacks_suppress=[click])


@click.command()
@click.argument("configuration_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--mode",
    type=click.Choice(
        [
            "mojave",
            "jet",
            "survey",
            "ppdisk",
        ],
        case_sensitive=False,
    ),
    default="mojave",
)
def main(configuration_path, mode) -> None:
    """Main simulation CLI tool for radiosim. Creates
    a dataset from a config file for a specified simulation
    mode.
    """
    config = Config.from_toml(configuration_path)

    LOGGER.info("Starting simulation of radio sky distributions:")
    LOGGER.info(pretty_repr(config))

    if config.general.seed is not None:
        LOGGER.info(f"Using numpy random seed {config.general.seed}.")
        np.random.seed(config.general.seed)

    outpath = config.paths.outpath
    sim_sources = check_outpath(
        outpath,
        verbose=config.general.verbose,
    )

    if mode in ["mojave", "jet", "survey"] and sim_sources:
        simulate_sky_distributions(config, mode)
    elif mode == "ppdisk":
        simulate_ppdisks(config)


if __name__ == "__main__":
    main()
