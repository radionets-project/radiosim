import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


def configure_axes(
    fig: matplotlib.figure.Figure | None,
    ax: matplotlib.axes.Axes | None,
    fig_args: dict = None,
):
    """Configures figure and axis depending if they were given
    as parameters.

    If neither figure nor axis are given, a new subplot will be created.
    If they are given the given ones will be returned.
    If only one of both is not given, this will cause an exception.

    Parameters
    ----------
    fig : matplotlib.figure.Figure | None
        The figure object.
    ax : matplotlib.axes.Axes | None
        The axes object.
    fig_args : dict, optional
        Optional arguments to be supplied to the ``plt.subplots`` call.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    if fig_args is None:
        fig_args = {}

    if None in (fig, ax) and not all(x is None for x in (fig, ax)):
        raise KeyError("The parameters ax and fig have to be both None or not None!")

    if ax is None:
        fig, ax = plt.subplots(layout="tight", **fig_args)

    return fig, ax


def get_norm(
    norm: str,
    vmax: float | None = None,
    vmin: float | None = None,
    vcenter: float = 0,
):
    """Converts a string parameter to a matplotlib norm.

    Parameters
    ----------
    norm : str
        The name of the norm.
        Possible values are:

        - ``log``:          Returns a logarithmic norm with clipping on (!), meaning
                            values above the maximum will be mapped to the maximum and
                            values below the minimum will be mapped to the minimum, thus
                            avoiding the appearance of a colormaps 'over' and 'under'
                            colors (e.g. in case of negative values). Depending on the
                            use case this is desirable but in case that it is not, one
                            can set the norm to ``log_noclip`` or provide a custom norm.

        - ``log_noclip``:   Returns a logarithmic norm with clipping off.

        - ``centered``:     Returns a linear norm which centered around zero.

        - ``sqrt``:         Returns a power norm with exponent 0.5, meaning the
                            square-root of the values.

        - other:            A value not declared above will be returned as is, meaning
                            that this could be any value which exists in matplotlib
                            itself.

    vmax : float | None, optional
        The maximum value of the range to normalize. This might not have an effect
        for every norm. Default is ``None``.

    vmin : float | None, optional
        The minimum value of the range to normalize. This might not have an effect
        for every norm. Default is ``None``.

    vcenter : float | None, optional
        The central value of the range to normalize. This might not have an effect
        for every norm. Default is ``0``.

    Returns
    -------
    matplotlib.colors.Normalize | str
        The norm or the str if no specific norm is defined for the string.
    """
    match norm:
        case "log":
            if vmin == 0:
                vmin = np.finfo(float).eps
                warnings.warn(
                    f"Since the given vmin is 0, the value was set to {vmin}"
                    " to enable logarithmic normalization.",
                    stacklevel=1,
                )

            return matplotlib.colors.LogNorm(clip=True, vmin=vmin, vmax=vmax)
        case "log_noclip":
            if vmin == 0:
                vmin = np.finfo(float).eps
                warnings.warn(
                    f"Since the given vmin is 0, the value was set to {vmin}"
                    " to enable logarithmic normalization.",
                    stacklevel=1,
                )

            return matplotlib.colors.LogNorm(clip=False, vmin=vmin, vmax=vmax)
        case "centered":
            if vmin is not None and vmax is not None:
                return matplotlib.colors.CenteredNorm(
                    vcenter=vcenter, halfrange=np.max([np.abs(vmin), np.abs(vmax)])
                )
            else:
                return matplotlib.colors.CenteredNorm(vcenter=vcenter)

        case "sqrt":
            return matplotlib.colors.PowerNorm(0.5, vmin=vmin, vmax=vmax)
        case _:
            return norm


def apply_crop(ax: matplotlib.axes.Axes, crop: tuple[list[float | None]]):
    """Applies a specific x and y limit ('crop') to the given axis.
    This will effectively crop the image.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis which to apply the limits to.
    crop : tuple[list[float | None]]
        The crop of the image. This has to have the format
        ``([x_left, x_right], [y_left, y_right])``, where the left and right
        values for each axis are the upper and lower limits of the axes which
        should be shown.
        IMPORTANT: If one supplies the ``plt.imshow`` an ``extent`` parameter,
        this will be the scale in which one has to give the crop! If not, the crop
        has to be in pixels.
    """
    ax.set_xlim(crop[0][0], crop[0][1])
    ax.set_ylim(crop[1][0], crop[1][1])


# based on https://stackoverflow.com/a/18195921 by "bogatron"
# Marked code (inside >>> BEGIN / <<< END) is licensed under CC BY-SA 3.0
def configure_colorbar(
    mappable: matplotlib.cm.ScalarMappable,
    ax: matplotlib.axes.Axes,
    fig: matplotlib.figure.Figure,
    label: str | None,
    show_ticks: bool = True,
    fontsize: str = "medium",
) -> matplotlib.colorbar.Colorbar:
    # >>> BEGIN
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(label, fontsize=fontsize)

    if not show_ticks:
        cbar.set_ticks([])
        cbar.ax.yaxis.set_major_formatter(NullFormatter())
        cbar.ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        cbar.ax.tick_params(labelsize=fontsize)
    # <<< END

    return cbar


def ellipse2cartesian(r: np.ndarray, phi: np.ndarray, a: float, b: float, alpha: float):
    alpha = np.deg2rad(alpha)
    return (
        r * (a * np.cos(phi) * np.cos(alpha) - b * np.sin(phi) * np.sin(alpha)),
        r * (a * np.cos(phi) * np.sin(alpha) + b * np.sin(phi) * np.cos(alpha)),
    )


def xy2pix(
    x: np.ndarray,
    y: np.ndarray,
    shape: tuple[int],
    xy_lims: tuple[list[float]] = ([-1, 1], [-1, 1]),
):
    xy_lims = np.array(xy_lims)

    delta_x = (np.abs(np.diff(xy_lims[0])) / shape[1])[0]
    delta_y = (np.abs(np.diff(xy_lims[1])) / shape[0])[0]

    col_idx = np.int64(np.floor((x - xy_lims[0, 0]) // delta_x))
    row_idx = np.int64(np.floor((y - xy_lims[1, 0]) // delta_y))

    return row_idx, col_idx


def ellipse_img2cartesian_img(
    r: np.ndarray,
    phi: np.ndarray,
    intensities: np.ndarray,
    grid_shape: tuple[int],
    a: float,
    b: float,
    alpha: float,
    dtype: type = np.float64,
    xy_lims: tuple[list[float]] = ([-1, 1], [-1, 1]),
):
    image = np.zeros(grid_shape, dtype=dtype)

    x, y = ellipse2cartesian(r, phi, a=a, b=b, alpha=alpha)
    row, col = xy2pix(x=x, y=y, shape=grid_shape, xy_lims=xy_lims)

    row_mask = np.logical_and(row < grid_shape[0], row > 0)
    col_mask = np.logical_and(col < grid_shape[1], col > 0)
    mask = np.logical_and(row_mask, col_mask)

    row = row[mask]
    col = col[mask]

    image[row, col] = intensities[mask]
    return image
