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
    xy_unit: un.Unit = un.AU,
    dtype: type = np.float64,
    cmap: str | matplotlib.colors.Colormap = "inferno",
    norm: str | matplotlib.colors.Normalize | None = None,
    intensity_limits: tuple | None = None,
    fig_args: dict | None = None,
    plot_args: dict | None = None,
    save_to: str | None = None,
    save_args: dict = None,
    fig: matplotlib.figure.Figure | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> tuple[matplotlib.image.AxesImage, matplotlib.figure.Figure, matplotlib.axes.Axes]:
    norm = get_norm(norm) if isinstance(norm, str) else norm

    intensity_label = "Intensity / a.u." if intensity_label is None else intensity_label

    save_args = {} if save_args is None else save_args

    plot_args = {} if plot_args is None else plot_args
    fig_args = {} if fig_args is None else fig_args

    fig, ax = configure_axes(fig=fig, ax=ax, fig_args=fig_args)

    im = ax.imshow(
        data,
        origin="lower",
        cmap=cmap,
        interpolation="none",
        norm=get_norm(norm=norm)
        if intensity_limits is None
        else get_norm(norm=norm, vmin=intensity_limits[0], vmax=intensity_limits[1]),
        extent=np.ravel(xy_lims) if xy_lims is not None else None,
        **plot_args,
    )

    configure_colorbar(mappable=im, ax=ax, fig=fig, label=intensity_label)

    ax.set_xlabel(f"$x$ / {xy_unit}")
    ax.set_ylabel(f"$y$ / {xy_unit}")

    if save_to is not None:
        fig.savefig(save_to, **save_args)

    return im, fig, ax
