import astropy.units as un
import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit


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


# For regular Keplerian motion
# Can be derived from eq. 4 from https://doi.org/10.1088/1538-3873/ae05cb
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
    interpolate_height: bool = True,
    interpolation_idx_extend: int = 10,
    min_height: un.Quantity | float = (
        0.5 * un.AU
    ),  # arbitrary choice to avoid too small values
) -> float | ArrayLike:
    val = (
        aspect_ratio(
            radius=radius,
            ref_aspect_ratio=ref_aspect_ratio,
            flaring_index=flaring_index,
            R0=R0,
        )
        * radius
    )

    min_height = (
        min_height.to(un.meter).value
        if isinstance(min_height, un.Quantity)
        else min_height
    )

    if min_height > 0 and np.max(val) > min_height and np.any(val < min_height):
        invalid_idx = np.argwhere(val > min_height)[0][0]

        # TODO: This is not really elegant, should be replaced by a better approach

        if interpolate_height:
            val[: invalid_idx + interpolation_idx_extend] = PchipInterpolator(
                np.append(
                    np.array([radius[0]]),
                    radius[invalid_idx + interpolation_idx_extend :],
                ),
                np.append(
                    np.array([min_height]),
                    val[invalid_idx + interpolation_idx_extend :],
                ),
            )(radius)[: invalid_idx + interpolation_idx_extend]
        else:
            val[:invalid_idx] = min_height
    return val


# Equation 14 in https://www.aanda.org/articles/aa/pdf/2010/05/aa13731-09.pdf
def approximate_grain_size(
    stokes_number: float,
    solid_dust_density: un.Quantity,
    gas_surface_density: un.Quantity,
) -> un.Quantity:
    return 2 * stokes_number * gas_surface_density / (np.pi * solid_dust_density)


def schmidt_number(stokes_number: float) -> float:
    # Fitted values according to https://doi.org/10.1051/0004-6361/200811220 p. 607
    schmidt_numbers = np.array([0.03, 0.4, 1.5])
    stokes_scale = np.array([1e-4, 1e-3, 1e-2])

    s_m, s_b = curve_fit(lambda x, m, b: m * x + b, stokes_scale, schmidt_numbers)[0]

    return s_m * stokes_number + s_b


# Diffusion coefficient definition from
# https://doi.org/10.1051/0004-6361/200811220 p. 600 eq. 20
def diffusion_coefficient(stokes_number: float, alpha_viscosity: float) -> float:
    return alpha_viscosity / schmidt_number(stokes_number=stokes_number)
