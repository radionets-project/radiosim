import click
from tqdm import tqdm
from radiosim.utils import (
    create_grid,
    add_noise,
    save_sky_distribution_bundle,
)
from radiosim.jet import create_jet
from radiosim.survey import create_survey


def simulate_sky_distributions(conf):
    for opt in ["train", "valid", "test"]:
        create_sky_distribution(
            conf=conf,
            opt=opt,
        )


def create_sky_distribution(conf, opt: str):
    for _ in tqdm(range(conf["bundles_" + opt])):
        grid = create_grid(conf["img_size"], conf["bundle_size"])
        if conf["mode"] == "jet":
            sky, target = create_jet(grid, conf)
        elif conf["mode"] == "survey":
            sky, target = create_survey(grid, conf)
        else:
            click.echo("Given mode not found. Choose 'survey' or 'jet' in config file")

        sky_bundle = sky.copy()
        target_bundle = target.copy()
        if conf["noise"] and conf["noise_level"] > 0:
            sky_bundle = add_noise(sky_bundle, conf["noise_level"])
            for img in sky_bundle:
                img -= img.min()
                img /= img.max()

        save_sky_distribution_bundle(conf, opt, sky_bundle, target_bundle)
