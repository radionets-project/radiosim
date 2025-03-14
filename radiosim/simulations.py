import click
import numpy as np
from numpy.random import default_rng
from tqdm import tqdm
from radiosim.utils import (
    create_grid,
    add_noise,
    adjust_outpath,
    save_sky_distribution_bundle,
    _save_mojave_bundle,
)
from radiosim.jet import create_jet
from radiosim.survey import create_survey
from radiosim.mojave import create_mojave


def simulate_sky_distributions(conf):
    if conf["mode"] == "mojave":
        seed = conf["seed"]
        if seed == "none":
            rng = default_rng()
        elif isinstance(seed, int):
            rng = default_rng(seed)
        else:
            raise TypeError('seed has to be int or "none"')
    else:
        rng = None
    for opt in ["train", "valid", "test"]:
        create_sky_distribution(
            conf=conf,
            opt=opt,
            rng=rng,
        )


def create_sky_distribution(conf, opt: str, rng=None) -> None:
    for _ in tqdm(range(conf["bundles_" + opt])):
        path = adjust_outpath(conf["outpath"], f"/data_{conf['mode']}_" + opt)
        grid = create_grid(conf["img_size"], conf["bundle_size"])
        if conf["mode"] == "jet":
            sky, target = create_jet(grid, conf)
        elif conf["mode"] == "survey":
            sky, target = create_survey(grid, conf)
        elif conf["mode"] == "mojave":
            data, data_name = create_mojave(conf, rng)
            sky_bundle = data[0].copy()
            if conf["noise"]:
                sky_bundle = np.squeeze(add_noise(np.expand_dims(
                    sky_bundle, axis=1), conf["noise_level"]), axis=1)
                # for img in data:
                #     img -= img.min()
                #     img /= img.max()
            _save_mojave_bundle(
                path, data=[sky_bundle, *data[1:]], data_name=data_name)
            continue
        else:
            click.echo(
                "Given mode not found. Choose 'survey', 'jet' or 'mojave' in config file")

        sky_bundle = sky.copy()
        target_bundle = target.copy()
        if conf["noise"] and conf["noise_level"] > 0:
            sky_bundle = add_noise(sky_bundle, conf["noise_level"])
            for img in sky_bundle:
                img -= img.min()
                img /= img.max()

        save_sky_distribution_bundle(path, sky_bundle, target_bundle)
