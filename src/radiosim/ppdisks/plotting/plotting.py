import astropy.units as un
import matplotlib
import numpy as np
from numpy.typing import ArrayLike

from .utils import (
    configure_axes,
    configure_colorbar,
    ellipse_img2cartesian_img,
    get_norm,
)


def plot_polar_image(
    polar_intensities: np.ndarray,
    grid_shape: tuple,
    r_lims: ArrayLike,
    phi_lims: ArrayLike | None = None,
    intensity_unit: str | None = None,
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
    if phi_lims is None:
        phi_lims = [-np.pi, np.pi]

    if xy_lims is None:
        xy_lims = ([-r_lims[1], r_lims[1]], [-r_lims[1], r_lims[1]])

    norm = get_norm(norm) if isinstance(norm, str) else norm

    intensity_unit = "Intensity / a.u." if intensity_unit is None else intensity_unit

    save_args = {} if save_args is None else save_args

    plot_args = {} if plot_args is None else plot_args
    fig_args = {} if fig_args is None else fig_args

    r = np.linspace(
        r_lims[0],
        r_lims[1],
        polar_intensities.shape[0],
    )
    phi = np.linspace(phi_lims[0], phi_lims[1], polar_intensities.shape[1])

    rs, phis = np.meshgrid(r, phi)
    rs = rs.T
    phis = phis.T

    data_trafo = ellipse_img2cartesian_img(
        r=rs,
        phi=phis,
        intensities=polar_intensities,
        grid_shape=grid_shape,
        a=a_maj,
        b=b_min,
        alpha=rot_angle,
        xy_lims=xy_lims,
    )

    fig, ax = configure_axes(fig=fig, ax=ax, fig_args=fig_args)

    im = ax.imshow(
        data_trafo,
        origin="lower",
        cmap=cmap,
        interpolation="none",
        norm=get_norm(norm=norm)
        if intensity_limits is None
        else get_norm(norm=norm, vmin=intensity_limits[0], vmax=intensity_limits[1]),
        extent=np.ravel(xy_lims),
        **plot_args,
    )

    configure_colorbar(mappable=im, ax=ax, fig=fig, label=intensity_unit)

    ax.set_xlabel(f"$x$ / {xy_unit}")
    ax.set_ylabel(f"$y$ / {xy_unit}")

    if save_to is not None:
        fig.savefig(save_to, **save_args)

    return im, fig, ax
