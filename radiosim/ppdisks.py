import numpy as np

import torch
from torch.fft import fft2, ifft2

from tqdm.autonotebook import tqdm

import matplotlib.pyplot as plt
from astropy.convolution import Gaussian2DKernel

from torchvision.transforms.functional import resize

from astropy.io import fits

import json
from pathlib import Path
import h5py

# Original code by Andreas Maisinger (TU Dortmund University)

def generate_proto_set(img_size: int, 
                       size: int, 
                       alpha_range: tuple = (0, 180), 
                       ratio_range: tuple = (3, 15), 
                       size_ratio_range: tuple = (0.1, 1), 
                       device: str = "cpu", 
                       seed: int = 1337):
    """
    Generates a Set of simulated protoplanetary disk images

    Parameters
    ----------
    img_size: int
    The size of the image (height and width)

    size: int
    The amount of images to simulate

    alpha_range: tuple of float, optional
    The range of values of the inclination of the disk

    ratio_range: tuple of float, optional
    The range of values of the ratio the minor axis should be of the major axis a / b

    size_ratio_range: tuple of float, optional
    The range of values of the ratio of the image size the disk should take up

    device: str, optional
    The name of the device to run the simulations on (with torch)

    seed: int, optional
    The seed for the random generator

    Returns
    -------
    protos: list of torch.tensor
    List of the simulated images

    params: dict
    The used simulation parameters
    
    """
    
    np.random.seed(seed)
    
    outputs = [
        create_proto(
        img_size, 
        alpha=np.random.randint(*alpha_range), 
        ratio=np.random.randint(*ratio_range) / np.random.randint(*ratio_range),
        size_ratio=np.random.uniform(*size_ratio_range),
        device=device
        )           
        for i in tqdm(range(size))
    ]
    
    protos = [out.cpu().numpy() for out in outputs]
    metadata = []
    params = dict(
        seed=seed,
        N=size,
        img_size=img_size,
        alpha_range=alpha_range,
        ratio_range=ratio_range,
        size_ratio_range=size_ratio_range
    )

    return protos, params

def create_proto(img_size: int, alpha: float, ratio: float, size_ratio: float, device: str = "cpu", seed: int = None):
    """
    Generates a simulated protoplanetary disk image

    Parameters
    ----------
    img_size: int
    The size of the image (height and width)

    alpha: float
    The inclination angle of the disk

    ratio: float
    The ratio the minor axis should be of the major axis a / b

    size_ratio: float
    The ratio of the image size the disk should take up

    device: str, optional
    The name of the device to run the simulations on (with torch)

    seed: int, optional
    The seed for the random generator. If set to ``None``, the seed is ignored.

    Returns
    -------
    proto: torch.tensor
    The simulated image of the protoplanetary disk
    
    """

    if seed is not None:
        np.random.seed(seed)
    
    fix_size = 900
    center = (fix_size // 2, fix_size // 2)
    m, n, results = _get_m_n("./mn/")
    core, d_core = _sim_core(fix_size, alpha, center, ratio, device=device)
    sc = np.random.randint(5,10)
    core = _smooth(sc, sc, core, device=device) * _get_scale(d_core, m, n)
    num_circ = int(np.random.normal(3, 0.5) // 1)

    els_full = torch.zeros((900, 900), device=device)
    ds = []
    a_max = d_core
    
    for i in range(num_circ):
        eli, di, maxlength = _sim_ring(
            fix_size,
            alpha,
            center,
            ratio,
            m,
            n,
            (i+1)*30,
            device=device
        )
        a_max = np.max([maxlength, a_max])
        els_full += eli
        ds.append(di)
    
    proto = core + els_full
    d = [d_core] + ds
    proto = torch.abs(proto)
    
    img_size2 = int(fix_size // 2)

    a_max = np.min([img_size2, a_max])
    
    proto_snip = proto[int(img_size2 - a_max) : int(img_size2 + a_max), 
                       int(img_size2 - a_max) : int(img_size2 + a_max)]

    if (proto_snip.max() < 1e-10):
        return create_proto(img_size, alpha, ratio, size_ratio)

    new_size = int(size_ratio * img_size)
    new_size = new_size if new_size % 2 == 0 else new_size - 1
    
    proto_snip = proto_snip[None, :, :]
    
    proto_snip = resize(proto_snip, (new_size, new_size))
    proto_snip = proto_snip[0]

    proto_snip_size2 = int(proto_snip.shape[0] // 2)

    img_size2 = int(img_size // 2)

    proto = torch.ones((img_size, img_size), device=device) * 1e-10
    proto[img_size2 - proto_snip_size2 : img_size2 + proto_snip_size2, 
          img_size2 - proto_snip_size2 : img_size2 + proto_snip_size2] += proto_snip

    del proto_snip    
    return proto.cpu()

def _create_ellipse(img_size: int, a: float, b: float, center: tuple, alpha: float, device: str = "cpu"):
    """
    Generates an ellipse

    Parameters
    ----------
    img_size: int
    The size of the image (height and width)

    a: float
    The major axis of the ellipse
    
    b: float
    The minor axis of the ellipse

    center: tuple of float
    The center of the ellipse

    alpha: float
    The rotation angle of the ellipse

    device: str, optional
    The name of the device to run the simulations on (with torch)

    Returns
    -------
    ellipse: torch.tensor
    The image of the ellipse

    average_distance: float
    The average distance from the center of the ellipse
    
    """
    
    x_0 = center[0]
    y_0 = center[1]
    matrix = torch.tensor([[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]], device=device)
    xx, yy = torch.from_numpy(np.mgrid[:img_size, :img_size]).to(device)
    term1 =( ((xx - x_0) * np.cos(np.deg2rad(alpha))) + (
        (yy - y_0) * np.sin(np.deg2rad(alpha))))**2
    term2 = (((xx - x_0) * np.sin(np.deg2rad(alpha))) - (
        (yy - y_0) * np.cos(np.deg2rad(alpha))))**2
    ellipse = ((term1 / a**2) + (term2 / b**2)) <= 1
    avarage_distance = (a + b)/2
    return ellipse, avarage_distance

def _substract(ellipse1: torch.tensor, ellipse2: torch.tensor, distance1: float, distance2: float):
    """
    Substracts two ellipses from eachother

    Parameters
    ----------
    ellipse1: torch.tensor
    The first ellipse
    
    ellipse2: torch.tensor
    The second ellipse

    distance1: float
    The average distance from the center of the first ellipse

    distance1: float
    The average distance from the center of the second ellipse

    Returns
    -------
    subtracted: torch.tensor
    The result of the subtraction

    average_distance: float
    The average distance from the center of the ellipse resulting from the subtraction
    
    """
    
    subtracted = ellipse1 ^ ellipse2
    average_distance = (distance1 + distance2) / 2
    return subtracted, average_distance

def _sim_core(img_size: int, alpha: float, center: tuple, ratio: float, device: str = "cpu"):
    """
    Simulates the core of the protoplanetary disk

    img_size: int
    The size of the image (height and width)

    alpha: float
    The rotation angle of the core

    center: tuple of float
    The center of the core

    ratio: float
    The ratio the minor axis should be of the major axis a / b

    device: str, optional
    The name of the device to run the simulations on (with torch)

    Returns
    -------
    core: torch.tensor
    The image of the core

    d_core: float
    The average distance from the center of the core
    
    """
    
    a = np.random.randint(3,15)
    core, d_core = _create_ellipse(img_size, a, a/ratio, center, alpha, device=device)
    return core, d_core

def _sim_ring(img_size: int, alpha: float, center: tuple, ratio: float, m: float, n: float, initial_a: float = 0, device: str = "cpu"):
    """
    Simulates a ring of the protoplanetary disk

    img_size: int
    The size of the image (height and width)

    alpha: float
    The rotation angle of the core

    center: tuple of float
    The center of the core

    ratio: float
    The ratio the minor axis should be of the major axis a / b

    m: float
    Empirical value determining the scale of the ring
    
    n: float
    Empirical value determining the scale of the ring

    initial_a: float, optional
    The initial size of the major axis

    device: str, optional
    The name of the device to run the simulations on (with torch)

    Returns
    -------
    ring: torch.tensor
    The image of the ring

    d_ring: float
    The average distance from the center of the ring

    max_value: float
    Largest extent of the ring
    
    """
    
    a = np.random.randint(10,35) + initial_a
    exp = np.random.randint(5,10)
    
    ring_a, d_ring_a = _create_ellipse(img_size, a, a/ratio, center, alpha, device=device)
    
    ring_b, d_ring_b = _create_ellipse(img_size, a+exp, (a+exp)/ratio, center, alpha, device=device)
    ring, d_ring = _substract(ring_b, ring_a, d_ring_b, d_ring_a)
    
    s = _get_smooth_factor(a)
    ring = _smooth(s, s, ring, device=device) * _get_scale(d_ring, m, n)
    
    max_value = np.max([a, a + exp, a / ratio, (a + exp) / ratio])
        
    return ring, d_ring, max_value + 6 * s

def _get_smooth_factor(a: float):
    """
    Empirically determines the standard deviation for the gaussian smoothing.
    """
    
    if a < 15:
        return 2
    if a < 20:
        return 4
    if a < 30:
        return 6
    if a < 40:
        return 9
    if a >= 40:
        return 12

# from https://stackoverflow.com/a/47979802
def _convolve2d(x: torch.tensor, y: torch.tensor):
    """
    Convolves two 2-dimensional images with eachother

    Parameters
    ----------
    x: torch.tensor
    The first image

    y: torch.tensor
    The second image

    Returns
    -------
    cc: torch.tensor
    The convolved image
    
    """
    
    fr = fft2(x)
    fr2 = fft2(torch.flipud(torch.fliplr(y)))
    m,n = fr.shape
    cc = torch.real(ifft2(fr*fr2))
    cc = torch.roll(cc, -int(m / 2) + 1, dims=0)
    cc = torch.roll(cc, -int(n / 2) + 1, dims=1)
    return cc

def _smooth(x: float, y: float, c: torch.tensor, device: str = "cpu"):
    """
    Smooths the input image by convolving it with a gaussian 2d-kernel

    Parameters
    ----------
    x: float
    The standard deviation in x-direction

    y: float
    The standard deviation in y-direction

    c: torch.tensor
    The image to smooth

    device: str, optional
    The name of the device to run the simulations on (with torch)
    
    """
    
    gauss_kernel = torch.from_numpy(Gaussian2DKernel(x_stddev = x, 
                                                     y_stddev = y, 
                                                     x_size=c.shape[0], 
                                                     y_size=c.shape[1])
                                    .array).to(device)
    return _convolve2d(c, gauss_kernel)
    
def _e(x: float, m: float, n: float):
    """
    Computes the value of an exponential function exp(m * x + n)
    """
    
    return np.exp(m * x + n)

def _get_m_n(path: str):
    """
    Get the empirical values for m and n from JSON files

    Parameters
    ----------
    path: str
    The path to the folder containing the files
    
    """
    
    path = Path(path)
    dict_paths = np.array([x for x in path.iterdir()])
    dicts = [json.load( open( d_path ) ) for d_path in dict_paths]
    results = {k: v for d in dicts for k, v in d.items()}
    id = np.random.choice(list(results.keys()))
    return results[id]["m"], results[id]["n"], results

def _get_scale(distance: float, m: float, n: float, noise_scale: float = 0.05):
    """
    Get the empirical standard deviation for the smoothing
    """
    
    return _e(distance, m, n) + np.random.normal(0, _e(distance, m, n)*noise_scale)