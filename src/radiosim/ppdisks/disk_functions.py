import astropy.units as un
import numpy as np
from numpy.typing import ArrayLike


# See https://fargo3d.github.io/documentation/def_setups.html#parameters
def surface_density(
    radius: float | ArrayLike, R0: float, sigma0: float, sigma_slope: float
):
    return sigma0 * (radius / R0) ** (-sigma_slope)


# From https://doi.org/10.1093/mnrasl/slv105 p.75
def sigma0(
    ref_radius: float | ArrayLike, R0: float, mass: float, sigma_slope: float
) -> float | ArrayLike:
    return (
        (2 - sigma_slope)
        / (2 * np.pi * R0**2)
        * mass
        * ((ref_radius / R0) ** (2 - sigma_slope) - 1) ** (-1)
    )


# See https://doi.org/10.1093/mnrasl/slv105 p.75
def mass_function(
    radius: float | ArrayLike, sigma_slope: float, sigma0: float, R0: float
):
    return (
        2
        * np.pi
        / (2 - sigma_slope)
        * sigma0
        * R0**2
        * ((radius / R0) ** (2 - sigma_slope) - 1)
    )


def orbital_period(
    mass: un.Quantity | float, radius: un.Quantity | float, G: un.Quantity | float
):
    return np.sqrt((4 * np.pi**2 * radius**3) / (mass * G))


# From https://fargo3d.github.io/documentation/def_setups.html#parameters
def aspect_ratio(
    radius: float | ArrayLike, R0: float, ref_aspect_ratio: float, flaring_index: float
) -> float | ArrayLike:
    return ref_aspect_ratio * (radius / R0) ** flaring_index


# From https://fargo3d.github.io/documentation/def_setups.html#parameters
def disk_height(
    radius: float | ArrayLike,
    ref_aspect_ratio: float,
    flaring_index: float,
    R0: float,
) -> float | ArrayLike:
    return (
        aspect_ratio(
            radius=radius,
            ref_aspect_ratio=ref_aspect_ratio,
            flaring_index=flaring_index,
            R0=R0,
        )
        * radius
    )
