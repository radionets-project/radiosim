import numpy as np
from tqdm import tqdm
from radiosim.utils import (
    create_grid,
    add_noise,
    adjust_outpath,
    save_sky_distribution_bundle,
)
from radiosim.jet import create_jet
from radiosim.point import create_point_source_img


def simulate_sky_distributions(sim_conf):
    for opt in ["train", "valid", "test"]:
        create_sky_distribution(
            img_size=sim_conf["img_size"],
            bundle_size=sim_conf["bundle_size"],
            num_bundles=sim_conf["bundles_" + str(opt)],
            noise=sim_conf["noise"],
            noise_level=sim_conf["noise_level"],
            num_jet_comps=sim_conf["num_jet_components"],
            num_point_gauss=sim_conf["num_point_gauss"],
            outpath=sim_conf["outpath"],
            option=opt,
        )


def create_sky_distribution(
    num_bundles,
    img_size,
    bundle_size,
    noise,
    noise_level,
    num_jet_comps,
    num_point_gauss,
    outpath,
    option,
):
    for i in tqdm(range(num_bundles)):
        grid = create_grid(img_size, bundle_size)
        if num_jet_comps:
            grid, jet_comps, jet_list = create_jet(grid, num_jet_comps)
        if num_point_gauss:
            grid, points, point_list = create_point_source_img(grid, num_point_gauss)

        source_bundle = grid.copy()
        if num_jet_comps and num_point_gauss:
            comp_bundle = np.array(
                [
                    np.append(jet_comps[i], points[i], axis=0)
                    if points[i].size > 0
                    else jet_comps[i]
                    for i in range(jet_comps.shape[0])
                ],
            )
            list_bundle = np.array(
                [
                    np.append(jet_list[i], point_list[i], axis=0)
                    if point_list[i].size > 0
                    else jet_list[i]
                    for i in range(jet_list.shape[0])
                ],
                dtype=object,
            )
        if num_jet_comps and not num_point_gauss:
            print("not point")
            comp_bundle = jet_comps
            list_bundle = jet_list
        if num_point_gauss and not num_jet_comps:
            print("not jet")
            comp_bundle = points
            list_bundle = point_list

        if noise:
            source_bundle = add_noise(source_bundle, noise_level)

        path = adjust_outpath(outpath, "/source_bundle_" + option)
        save_sky_distribution_bundle(path, source_bundle, comp_bundle, list_bundle)
