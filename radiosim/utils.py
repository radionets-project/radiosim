import os
import sys
import h5py
import click
import numpy as np
from pathlib import Path


def create_grid(pixel, bundle_size):
    """
    Creates a square 2d grid.

    Parameters
    ----------
    pixel: int
        number of pixel in x and y

    Returns
    -------
    grid: ndarray
        2d grid with 1e-10 pixels, X meshgrid, Y meshgrid
    """
    x = np.linspace(0, pixel - 1, num=pixel)
    y = np.linspace(0, pixel - 1, num=pixel)
    X, Y = np.meshgrid(x, y)
    grid = np.array([np.zeros(X.shape) + 1e-10, X, Y])
    grid = np.repeat(
        grid[None, :, :, :],
        bundle_size,
        axis=0,
    )
    return grid


def get_exp(size=1):
    num = np.ceil(size / 2).astype(int)
    exp = np.random.exponential(scale=0.08, size=(num,))
    exp_inv = 1 - np.random.exponential(scale=0.08, size=(num,))
    vals = np.hstack([exp, exp_inv])
    return np.random.choice(vals, size=size)


def check_outpath(outpath, quiet=False):
    """
    Check if outpath exists. Check for existing source_bundle files.
    Ask to overwrite or reuse existing files.
    Parameters
    ----------
    outpath : str
        path to out directory
    quiet : bool
        activate quiet mode, overwrite existing files

    Returns
    -------
    sim_sources : bool
        flag to enable/disable source simulation routine
    """
    path = Path(outpath)
    exists = path.exists()
    if exists is True:
        source = {p for p in path.rglob("*source_bundle*.h5") if p.is_file()}
        if source:
            click.echo("Found existing source simulations!")
            if quiet:
                click.echo("Overwriting existing source simulations!")
                [p.unlink() for p in source]
                sim_sources = True
                return sim_sources
            elif click.confirm(
                "Do you really want to overwrite existing files?", abort=False
            ):
                click.echo("Overwriting existing source simulations!")
                [p.unlink() for p in source]
                sim_sources = True
                return sim_sources
            else:
                click.echo("Keeping existing source simulations!")
                sim_sources = False
                sys.exit()
        else:
            sim_sources = True
    else:
        Path(path).mkdir(parents=True, exist_ok=False)
        sim_sources = True
    return sim_sources


def read_config(config):
    sim_conf = {}
    sim_conf["outpath"] = config["paths"]["outpath"]
    if config["source_types"]["jets"]:
        click.echo("Adding jet sources to sky distributions! \n")
        sim_conf["num_jet_components"] = config["source_types"]["num_jet_components"]

    if config["source_types"]["pointlike_gaussians"]:
        click.echo("Adding poinhtlike Gaussians to sky distributions! \n")

    sim_conf["bundles_train"] = config["image_options"]["bundles_train"]
    sim_conf["bundles_valid"] = config["image_options"]["bundles_valid"]
    sim_conf["bundles_test"] = config["image_options"]["bundles_test"]
    sim_conf["bundle_size"] = config["image_options"]["bundle_size"]
    sim_conf["img_size"] = config["image_options"]["img_size"]
    sim_conf["noise"] = config["image_options"]["noise"]
    sim_conf["noise_level"] = config["image_options"]["noise_level"]
    return sim_conf


def get_noise(image, scale, mean=0, std=1):
    """
    Calculate random noise values for all image pixels.
    Parameters
    ----------
    image: 2darray
        2d image
    scale: float
        scaling factor to increase noise
    mean: float
        mean of noise values
    std: float
        standard deviation of noise values
    Returns
    -------
    out: ndarray
        array with noise values in image shape
    """
    return np.random.normal(mean, std, size=image.shape) * scale


def add_noise(bundle, noise_level):
    """
    Used for adding noise and plotting the original and noised picture,
    if asked. Using 0.05 * max(image) as scaling factor.
    Parameters
    ----------
    bundle: path
        path to hdf5 bundle file
    noise_level: int
        noise level in percent
    Returns
    -------
    bundle_noised hdf5_file
        bundle with noised images
    """
    bundle_noised = np.array(
        [img + get_noise(img, (img.max() * noise_level / 100)) for img in bundle]
    )
    return bundle_noised


def adjust_outpath(path, option, form="h5"):
    """
    Add number to out path when filename already exists.
    Parameters
    ----------
    path: str
        path to save directory
    option: str
        additional keyword to add to path
    Returns
    -------
    out: str
        adjusted path
    """
    counter = 0
    filename = str(path) + (option + "{}." + form)
    while os.path.isfile(filename.format(counter)):
        counter += 1
    out = filename.format(counter)
    return out


def save_sky_distribution_bundle(
    path, x, y, z=None, name_x="x", name_y="y", name_z="list"
):
    """
    write fft_pairs created in second analysis step to h5 file
    """
    with h5py.File(path, "w") as hf:
        hf.create_dataset(name_x, data=x)
        hf.create_dataset(name_y, data=y)
        if z is not None:
            hf.create_dataset(name_z, data=z)
        hf.close()
