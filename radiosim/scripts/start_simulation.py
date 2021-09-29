import toml
import click
from radiosim.utils import read_config
from radiosim.simulations import simulate_sky_distributions


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

    if mode == "simulate":
        simulate_sky_distributions(sim_conf)

    # if mode == "overview":
    #     create_simulation_overview(sim_conf)


if __name__ == "__main__":
    main()
