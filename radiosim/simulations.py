import click
from tqdm import tqdm
from radiosim.utils import (
    create_grid,
    add_noise,
    adjust_outpath,
    save_sky_distribution_bundle,
)
from radiosim.jet import create_jet
from radiosim.survey import create_survey


def simulate_sky_distributions(sim_conf):
    for opt in ["train", "valid", "test"]:
        create_sky_distribution(
            mode=sim_conf["mode"],
            outpath=sim_conf["outpath"],
            train_type=sim_conf["training_type"],
            num_jet_comps=sim_conf["num_jet_components"],
            num_sources=sim_conf["num_sources"],
            class_distribution=sim_conf["class_distribution"],
            scale_sources=sim_conf["scale_sources"],
            num_bundles=sim_conf["bundles_" + str(opt)],
            bundle_size=sim_conf["bundle_size"],
            img_size=sim_conf["img_size"],
            noise=sim_conf["noise"],
            noise_level=sim_conf["noise_level"],
            option=opt,
        )


def create_sky_distribution(
    mode,
    outpath,
    train_type,
    num_jet_comps,
    num_sources,
    class_distribution,
    scale_sources,
    num_bundles,
    bundle_size,
    img_size,
    noise,
    noise_level,
    option,
):
    for i in tqdm(range(num_bundles)):
        grid = create_grid(img_size, bundle_size)
        if mode == "jet":
            sky, target = create_jet(grid, num_jet_comps, train_type)
        elif mode == "survey":
            sky, target = create_survey(
                grid, num_sources, class_distribution, scale_sources
            )
        else:
            click.echo("Given mode not found. Choose 'survey' or 'jet' in config file")
        sky_bundle = sky.copy()
        target_bundle = target.copy()
        if noise:
            sky_bundle = add_noise(sky_bundle, noise_level)
            for img in sky_bundle:
                img -= img.min()
                img /= img.max()
        path = adjust_outpath(outpath, "/samp_" + option)
        save_sky_distribution_bundle(
            path, sky_bundle, target_bundle
        )
