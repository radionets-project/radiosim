import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.nddata.utils import Cutout2D


def gauss(params: list, size: int):
    """
    Returns a 2d gaussian distribution. Creates a gaussian in the center with twice
    the image size and then cuts out the correct position.
    Pro: Easy to understand and maintain
    Con: Slower than twodgaussian; maybe faster, if the cutout could be removed

    Parameters
    ----------
    params: list
        [amplitude, center_x, center_y, width_x, width_y, rot]
        (rot (rotation) in radian)
    shape: int
        length of the image
    
    Returns
    -------
    gaussian_scaled: 2darray
        gaussian distribution in two dimensions
    """
    gaussian_2D_kernel = Gaussian2DKernel(
        x_stddev = params[3],
        y_stddev = params[4],
        theta = params[5],
        x_size = size * 2,
        y_size = size * 2,
        )
    gaussian_cutout = Cutout2D(
        data = gaussian_2D_kernel,
        position = (size * 1.5 - params[1], size * 1.5 - params[2]),
        size = (size, size),
        ).data
    gaussian_scaled = gaussian_cutout / gaussian_cutout.max() * params[0]
    return gaussian_scaled


def twodgaussian(params: list, size: int):
    """
    Returns a 2d gaussian function of the form:
    x' = np.cos(rot) * x - np.sin(rot) * y
    y' = np.sin(rot) * x + np.cos(rot) * y
    (rot should be in degrees)
    g = a * np.exp ( - ( ((x-center_x)/width_x)**2 +
    ((y-center_y)/width_y)**2 ) / 2 )

    Pro: Faster than gauss
    Con: More difficult to understand

    Parameters
    ----------
    params: list
        [amplitude, center_x, center_y, width_x, width_y, rot]
        (rot (rotation) in degrees)
    size: int
        length of the image
    
    Returns
    -------
    rotgauss: 2darray
        gaussian distribution in two dimensions
    
    Short version of 'twodgaussian':
    https://github.com/keflavich/gaussfitter/blob/0891cd3605ab5ba000c2d5e1300dd21c15eee1fd/gaussfitter/gaussfitter.py#L75
    """
    amplitude = params[0]
    center_y, center_x = params[1], params[2]
    width_y, width_x = params[3], params[4]
    rot = params[5]
    rot = np.pi/180 * rot
    rcen_x = center_x * np.cos(rot) - center_y * np.sin(rot)
    rcen_y = center_x * np.sin(rot) + center_y * np.cos(rot)

    def rotgauss(x, y):
        xp = x * np.cos(rot) - y * np.sin(rot)
        yp = x * np.sin(rot) + y * np.cos(rot)
        g = amplitude*np.exp(-(((rcen_x-xp)/width_x)**2 +
                                      ((rcen_y-yp)/width_y)**2)/2.)
        return g
    return rotgauss(*np.indices((size, size)))