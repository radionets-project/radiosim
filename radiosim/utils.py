import numpy as np
import click
from pathlib import Path
import sys


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
    exp_inv = 1 - np.random.exponential(scale=0.08, size=(num,))
    vals = np.hstack([exp, exp_inv])
    return np.random.choice(vals, size=size)


def check_outpath(outpath, quiet=False):
    """
    Check if outpath exists. Check for existing fft_files and sampled-files.
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
