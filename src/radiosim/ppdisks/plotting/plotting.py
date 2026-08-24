import astropy.units as un
import matplotlib
import numpy as np
from numpy.typing import ArrayLike

from .utils import (
    configure_axes,
    configure_colorbar,
    get_norm,
)


def plot_image(
    data: np.ndarray,
    intensity_label: str | None = None,
    a_maj: float = 1.0,
    b_min: float = 1.0,
    rot_angle: float = 0.0,
    xy_lims: ArrayLike | None = None,
    xy_labels: tuple[str] = ("$x$", "$y$"),
    xy_unit: un.Unit | tuple[un.Unit] = un.AU,
    cmap: str | matplotlib.colors.Colormap = "inferno",
    norm: str | matplotlib.colors.Normalize | None = None,
    intensity_limits: tuple | None = None,
    fig_args: dict | None = None,
    plot_args: dict | None = None,
    save_to: str | None = None,
    save_args: dict | None = None,
    fig: matplotlib.figure.Figure | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> tuple[matplotlib.image.AxesImage, matplotlib.figure.Figure, matplotlib.axes.Axes]:
    intensity_label = "Intensity / a.u." if intensity_label is None else intensity_label

    save_args = {} if save_args is None else save_args

    plot_args = {"origin": "lower"} if plot_args is None else plot_args
    fig_args = {} if fig_args is None else fig_args

    intensity_limits = [None, None] if intensity_limits is None else intensity_limits

    fig, ax = configure_axes(fig=fig, ax=ax, fig_args=fig_args)

    im = ax.imshow(
        data,
        cmap=cmap,
        norm=get_norm(norm=norm, vmin=intensity_limits[0], vmax=intensity_limits[1]),
        extent=np.ravel(xy_lims) if xy_lims is not None else None,
        **plot_args,
    )

    configure_colorbar(mappable=im, ax=ax, fig=fig, label=intensity_label)

    if not isinstance(xy_unit, tuple):
        xy_unit = (xy_unit, xy_unit)

    ax.set_xlabel(f"{xy_labels[0]} / {xy_unit[0]}")
    ax.set_ylabel(f"{xy_labels[1]} / {xy_unit[1]}")

    if save_to is not None:
        fig.savefig(save_to, **save_args)

    return im, fig, ax
