import os
import re
import sys
import cv2
import h5py
import toml
import click
import torch
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
    gamma = 1 / np.sqrt(1 - beta ** 2)  # Lorentz factor
    mu = np.cos(theta)

    boost_app = 1 / (gamma * (1 - beta * mu))
    boost_rec = 1 / (gamma * (1 + beta * mu))
    return boost_app, boost_rec


def zoom_on_source(img, comp=None, max_amp=0.01):
    """
    Zoom on source to cut out irrelevant area. Shape will stay equal.

    Parameters
    ----------
    img: 2D array
        Image of the sky used to zoom on
    comp: 3D array (n, (img))
        Images of the components, same zooming as on img
    max_amp: float
        Maximal amplitude which will be at the edge of the image

    Returns
    -------
    zoomed_img: ndarray
        Image after zooming
    zoom_factor: float
        Zooming factor
    """
    # find farest outside column or row with amplitude > max_amp
    mask = img > max_amp
    mask_flip = np.flip(mask)

    idx_left = np.argmax(np.sum(mask, axis=0) > 0)
    idx_right = np.argmax(np.sum(mask_flip, axis=0) > 0)
    idx_bottom = np.argmax(np.sum(mask, axis=1) > 0)
    idx_top = np.argmax(np.sum(mask_flip, axis=1) > 0)
    # print(idx_left, idx_right, idx_bottom, idx_top)
    idx = np.min([idx_left, idx_right, idx_bottom, idx_top])
    size = img.shape[0]
    zoom_factor = size / (size - 2 * idx)

    # crop the source
    cropped_img = img[idx:size-idx, idx:size-idx]
    zoomed_img = cv2.resize(cropped_img, dsize=(size, size), interpolation=cv2.INTER_LINEAR)

    if comp is not None:
        cropped_comp = comp[:, idx:size-idx, idx:size-idx]
        zoomed_comp = np.empty_like(comp)
        for i, component in enumerate(cropped_comp):
            zoomed_comp[i] = cv2.resize(component, dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        return zoomed_img, zoomed_comp, zoom_factor

    return zoomed_img, zoom_factor


def zoom_out(img, comp=None, pad_value=0):
    """
    Zoom out of an image. Padding edges with zeros.

    Parameters
    ----------
    img: 2D array
        Image of the sky
    comp: 3D array (n, (img))
        Images of the components
    pad_value: int
        Number of pixels to pad around the source

    Returns
    -------
    zoomed_img: ndarray
        Image after zooming
    zoomed_comp: ndarray
        Componets after zooming
    """
    if not isinstance(pad_value, int):
        pad_value = np.int64(pad_value)
    size = img.shape[0]
    img = cv2.resize(np.pad(img, pad_value), dsize=(size, size), interpolation=cv2.INTER_LINEAR)

    if comp is not None:
        for component in comp:
            component = cv2.resize(np.pad(component, pad_value), dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        return img, comp

    return img


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
    sim_conf["mode"] = config["general"]["mode"]
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


def adjust_outpath(path, option, form="h5"):
    """
    Add number to out path when filename already exists.

    Parameters
    ----------
    path: str
        path to save directory
    option: str
        additional keyword to add to path
    form: str
        file extension

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
    r = np.sqrt(x ** 2 + y ** 2)
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


from math import pi


def bmul(vec, mat, axis=0):
    """Expand vector for batchwise matrix multiplication.
    Parameters
    ----------
    vec : 2dtensor
        vector for multiplication
    mat : 3dtensor
        matrix for multiplication
    axis : int, optional
        batch axis, by default 0
    Returns
    -------
    3dtensor
        Product of matrix multiplication. (bs, n, m)
    """
    mat = mat.transpose(axis, -1)
    return (mat * vec.expand_as(mat)).transpose(axis, -1)


def pca(image):
    """
    Compute the major components of an image. The Image is treated as a
    distribution.
    Parameters
    ----------
    image: Image or 2DArray (N, M)
            Image to be used as distribution
    Returns
    -------
    cog_x: Skalar
            X-position of the distributions center of gravity
    cog_y: Skalar
            Y-position of the distributions center of gravity
    psi: Skalar
            Angle between first mjor component and x-axis
    """

    pix_x, pix_y, image = im_to_array_value(image)

    cog_x = (torch.sum(pix_x * image, axis=1) / torch.sum(image, axis=1)).unsqueeze(-1)
    cog_y = (torch.sum(pix_y * image, axis=1) / torch.sum(image, axis=1)).unsqueeze(-1)

    delta_x = pix_x - cog_x
    delta_y = pix_y - cog_y

    inp = torch.cat([delta_x.unsqueeze(1), delta_y.unsqueeze(1)], dim=1)

    cov_w = bmul(
        (cog_x - 1 * torch.sum(image * image, axis=1).unsqueeze(-1) / cog_x).squeeze(1),
        (torch.matmul(image.unsqueeze(1) * inp, inp.transpose(1, 2))),
    )

    eig_vals_torch, eig_vecs_torch = torch.linalg.eigh(cov_w, UPLO="U")

    psi_torch = torch.atan(eig_vecs_torch[:, 1, 1] / eig_vecs_torch[:, 0, 1])

    return cog_x, cog_y, psi_torch


def calc_jet_angle(image):
    """Caluclate the jet angle from an image created with gaussian sources. This
    is achieved by a PCA.

    Parameters
    ----------
    image : ndarray
        input image
    Returns
    -------
    float
        slope of the line
    float
        intercept of the line
    float
        angle between the horizontal axis and the jet axis
    """
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image)
    image = image.clone()
    img_size = image.shape[-1]
    # ignore negative pixels, which can appear in predictions
    image[image < 0] = 0

    if len(image.shape) == 2:
        image = image.unsqueeze(0)

    batch_size = image.shape[0]

    # Search for sources with two maxima
    maxima = []
    for i in range(image.shape[0]):
        a = torch.where(image[i] == image[i].max())
        if len(a[0]) > 1:
            # if two maxima are found, interpolate to the middle in x and y direction
            mid_x = (a[0][1] - a[0][0]) // 2 + a[0][0]
            mid_y = (a[1][1] - a[1][0]) // 2 + a[1][0]
            maxima.extend([(mid_x, mid_y)])
        else:
            maxima.extend([a])

    vals = torch.tensor(maxima)
    x_mid = vals[:, 0]
    y_mid = vals[:, 1]
    
    # only use jet structure
    for img in image:
        img[(img > img.max() * 0.4) | (img < img.max() * 0.15)] = 0

    _, _, alpha_pca = pca(image)

    m = torch.tan(pi / 2 - alpha_pca)
    n = y_mid - m * x_mid
    alpha = (alpha_pca) * 180 / pi
    return m, n, alpha


def im_to_array_value(image):
    """
    Transforms the image to an array of pixel coordinates and the containt
    intensity

    Parameters
    ----------
    image: Image or 2DArray (N, M)
            Image to be transformed

    Returns
    -------
    x_coords: Numpy 1Darray (N*M, 1)
            Contains the x-pixel-position of every pixel in the image
    y_coords: Numpy 1Darray (N*M, 1)
            Contains the y-pixel-position of every pixel in the image
    value: Numpy 1Darray (N*M, 1)
            Contains the image-value corresponding to every x-y-pair

    """
    num = image.shape[0]
    pix = image.shape[-1]

    a = torch.arange(0, pix, 1)
    grid_x, grid_y = torch.meshgrid(a, a, indexing="xy")
    x_coords = torch.cat(num * [grid_x.flatten().unsqueeze(0)])
    y_coords = torch.cat(num * [grid_y.flatten().unsqueeze(0)])
    value = image.reshape(-1, pix ** 2)
    return x_coords, y_coords, value

