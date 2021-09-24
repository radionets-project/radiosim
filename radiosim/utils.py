import numpy as np


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
    num = np.ceil(size / 2)
    exp = np.random.exponential(scale=0.08, size=(num,))
    exp_inv = (1 - np.random.exponential(scale=0.08, size=(num,)))
    vals = np.hstack([exp, exp_inv])
    return np.random.choice(vals, size=size)
