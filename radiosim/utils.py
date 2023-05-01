import os
import re
import sys
import cv2
import h5py
import toml
import click
import numpy as np
from pathlib import Path
from scipy import signal
from astropy.convolution import Gaussian2DKernel


def create_grid(pixel, bundle_size):
    """
    Creates a square 2d grid.

    Parameters
    ----------
    pixel: int
        number of pixel in x and y
    bundle_size: int
        number of images in each bundle

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


def relativistic_boosting(theta, beta):
    """
    Calculate relativistic boosting factor for a jet.

    Parameters
    ----------
    theta: float
        angle of the jet in relation to the observer
    beta: float
        velocity of the jet components

    Returns
    -------
    boost_app: float
        boosting factor for the approaching jet
    boost_rec: float
        boosting factor for the receding jet
    """
    gamma = 1 / np.sqrt(1 - beta**2)  # Lorentz factor
    mu = np.cos(theta)

    boost_app = 1 / (gamma * (1 - beta * mu))
    boost_rec = 1 / (gamma * (1 + beta * mu))
    return boost_app, boost_rec


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
    """
    Unpacking of the config file to print the config parameters

    Parameters
    ----------
    config: toml-file
        toml configuration file with all parameters
    Returns
    -------
    sim_comf: dictionary
        unpacked configurations
    """
    sim_conf = {}
    sim_conf["quiet"] = config["general"]["quiet"]
    sim_conf["mode"] = config["general"]["mode"]
    sim_conf["multiprocessing"] = config["general"]["multiprocessing"]
    sim_conf["outpath"] = config["paths"]["outpath"]
    sim_conf["training_type"] = config["jet"]["training_type"]
    sim_conf["num_jet_components"] = config["jet"]["num_jet_components"]
    sim_conf["scaling"] = config["jet"]["scaling"]
    sim_conf["num_sources"] = config["survey"]["num_sources"]
    sim_conf["class_distribution"] = config["survey"]["class_distribution"]
    sim_conf["scale_sources"] = config["survey"]["scale_sources"]
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
        maximum intensity of noise in percent

    Returns
    -------
    image_noised: 4darray
        bundle of noised images
    """

    def noise_small(kernel, mean=0, std=1):
        """
        Create the noise of different kernel sizes
        """
        max_noise = np.random.uniform(0, 1, img_shape[0])
        noise = (
            np.random.normal(mean, std, size=img_shape) * max_noise[:, None, None, None]
        )
        g_kernel = Gaussian2DKernel(kernel / 2).array[None, None, :]
        return signal.convolve(noise, g_kernel, mode="same")

    def call_noise(kernels, strengths):
        """
        Loop through lists and normalize
        """
        noise_out = np.zeros(shape=img_shape)
        for kernel, strength in zip(kernels, strengths):
            if kernel == 1:
                noise = np.random.normal(size=img_shape)
            else:
                noise = noise_small(kernel)
            noise /= np.abs(noise).max() / strength
            noise_out += noise
        return noise_out

    img_shape = image.shape
    kernels = [1, img_shape[-1] / 32, img_shape[-1] / 8]
    strengths = [0.2, 0.3, 0.5]  # have to add up to 1

    noise = call_noise(kernels, strengths)
    noise /= np.abs(noise).max() / (noise_level / 100)
    image_noised = image + noise

    return image_noised


def save_sky_distribution_bundle(path, x, y, name_x="x", name_y="y"):
    """
    Write images created in analysis to h5 file.

    Parameters
    ----------
    path: str
        path to save file
    x: ndarray
        image of the full jet, sum over all components
    y: ndarray
        images of components or list, depends on train_type
    name_x: str
        name of the x-data
    name_y: str
        name of the y-data
    """
    with h5py.File(path, "w") as hf:
        hf.create_dataset(name_x, data=x)
        hf.create_dataset(name_y, data=y)
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
        angle in radian
    """
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (r, phi)


def pol2cart(r: float, phi: float):
    """
    Transforms polar to cartesian coordinates.

     Parameters
    ----------
    r: float
        radius, euclidean distance between (0,0) and (x,y)
    phi: float
        angle in radian

    Returns
    -------
    x: float
        x-coordinate
    y: float
        y-coordinate
    """
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return (x, y)


def load_data(conf_path, data_type="train", key="x"):
    config = toml.load(conf_path)
    path = Path(config["paths"]["outpath"])
    bundle_paths = np.array([x for x in path.iterdir()])
    paths = [
        path for path in bundle_paths if re.findall("samp_" + data_type, path.name)
    ]
    data = []
    for path_test in paths:
        df = h5py.File(path_test, "r")
        data.extend(df[key])
    return data
