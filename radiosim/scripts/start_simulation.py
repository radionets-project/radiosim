import toml
import click
from radiosim.utils import read_config, check_outpath
from radiosim.simulations import simulate_sky_distributions
from radiosim.plotting import create_simulation_overview


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
def main(configuration_path, mode):
    config = toml.load(configuration_path)
    sim_conf = read_config(config)
    print(sim_conf, "\n")

    outpath = config["paths"]["outpath"]
    sim_sources = check_outpath(
        outpath,
        quiet=config["mode"]["quiet"],
    )

    if mode == "simulate":
        if sim_sources:
            simulate_sky_distributions(sim_conf)

    if mode == "overview":
        if sim_sources:
            simulate_sky_distributions(sim_conf)
        create_simulation_overview(sim_conf)


if __name__ == "__main__":
    main()
