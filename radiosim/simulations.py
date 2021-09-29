from tqdm import tqdm
from radiosim.utils import (
    create_grid,
    add_noise,
    adjust_outpath,
    save_sky_distribution_bundle,
)
from radiosim.jet import create_jet


def simulate_sky_distributions(sim_conf):
    for opt in ["train", "valid", "test"]:
        create_sky_distribution(
            img_size=sim_conf["img_size"],
            bundle_size=sim_conf["bundle_size"],
            num_bundles=sim_conf["bundles_" + str(opt)],
            noise=sim_conf["noise"],
            noise_level=sim_conf["noise_level"],
            outpath=sim_conf["outpath"],
            option=opt,
        )


def create_sky_distribution(
    num_bundles, img_size, bundle_size, noise, noise_level, outpath, option
):
    for i in tqdm(range(num_bundles)):
        grid = create_grid(img_size, bundle_size)
        jet, jet_comps, source_list = create_jet(grid)

        print(source_list)

        jet_bundle = jet.copy()
        comp_bundle = jet_comps.copy()

        if noise:
            jet_bundle = add_noise(jet_bundle, noise_level)

        path = adjust_outpath(outpath, "/source_bundle_" + option)
        save_sky_distribution_bundle(path, jet_bundle, comp_bundle, source_list)
