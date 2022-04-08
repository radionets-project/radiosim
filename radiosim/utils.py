import os
import sys
import h5py
import click
import torch
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
    """
    Returns random numbers between 0 and 1. The probability distribution looks like an 'U'.
    Used for the parameter 'alpha' to change the amplitude of the counter jet.

    Parameters
    ----------
    size: int
        quantity of random numbers to be returned

    Returns
    -------
    vals: ndarray
        array of random numbers
    """
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
        source = {p for p in path.rglob("*samp*.h5") if p.is_file()}
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
    sim_conf["training_type"] = config["mode"]["training_type"]
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


def add_noise(image, noise_level):
    """
    Used for adding noise.

    Parameters
    ----------
    image: 4darray
        bundle of images of shape [n, 1, size, size]
    noise_level: int
        noise level in percent

    Returns
    -------
    image_noised 4darray
        bundle of noised images
    """
    img_shape = image.shape

    def advanced_noise(scaling, mean=0, std=1):
        size_ratio = img_shape[-1] / scaling
        size_int = np.int(size_ratio)
        size_rescale = size_ratio / size_int * scaling
        size_noise = (img_shape[0], 1, size_int, size_int)
        max_noise = np.random.uniform(0, 1, img_shape[0])
        noise = np.random.normal(
            loc=mean,
            scale=std,
            size=size_noise
            ) * max_noise[:, None, None, None]
        noise = torch.nn.functional.interpolate(
            torch.tensor(noise),
            scale_factor=size_rescale,
            mode='bicubic',
            align_corners=True
            ).cpu().detach().numpy()
        return noise
    
    strengths = [0.2, 0.3, 0.5]  # have to add up to 1
    noise_small = np.random.normal(loc=0, scale=1, size=img_shape)
    noise_small /= noise_small.max() / strengths[0]
    noise_medium = advanced_noise(4)
    noise_medium /= noise_medium.max() / strengths[1]
    noise_large = advanced_noise(32)
    noise_large /= noise_large.max() / strengths[2]

    noise = noise_small + noise_medium + noise_large
    noise /= noise.max() / (noise_level / 100)
    image_noised = image + noise

    return image_noised


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
    filename = str(path) + (option + "_{}." + form)
    while os.path.isfile(filename.format(counter)):
        counter += 1
    out = filename.format(counter)
    return out


def save_sky_distribution_bundle(
    path, train_type, x, y, z=None, name_x="x", name_y="y", name_z="list"
):
    """
    write fft_pairs created in second analysis step to h5 file
    """
    with h5py.File(path, "w") as hf:
        hf.create_dataset(name_x, data=x)
        if train_type in ['gauss', 'clean']:
            hf.create_dataset(name_y, data=y)
            if z is not None:
                hf.create_dataset(name_z, data=z)     
        elif train_type == 'list':
            hf.create_dataset(name_y, data=z)
        elif train_type == 'counts':
            hf.create_dataset(name_y, data=y)
            hf.create_dataset(name_z, data=z)
        hf.close()


def cart2pol(x: float, y: float):
    """
    Transforms cartesian to polar coordinates.

    Parameters
    ----------
    x: float
        x-coordinate
    y: float
        y-coordinate

    Returns
    -------
    r: float
        radius, euclidean distance between (0,0) and (x,y)
    phi: float
        angle in degree
    """
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x) * 180 / np.pi
    return (r, phi)


def pol2cart(r: float, phi: float):
    """
    Transforms polar to cartesian coordinates.

     Parameters
    ----------
    r: float
        radius, euclidean distance between (0,0) and (x,y)
    phi: float
        angle in degree

    Returns
    -------
    x: float
        x-coordinate
    y: float
        y-coordinate
    """
    x = r * np.cos(phi / 180 * np.pi)
    y = r * np.sin(phi / 180 * np.pi)
    return (x, y)