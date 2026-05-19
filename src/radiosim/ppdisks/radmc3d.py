from dataclasses import dataclass
from pathlib import Path

import astropy.units as un
import numpy as np
from astropy.convolution import Gaussian2DKernel

from radiosim.ppdisks.simulation import DiskModel

from .config.radmc3d import format_output_lines


# FFT Convolution algorithmfrom https://stackoverflow.com/a/47979802
# Marked code (inside >>> BEGIN / <<< END) is licensed under CC BY-SA 3.0
def _convolve2d(x: np.ndarray, y: np.ndarray):
    # >>> BEGIN
    fr = np.fft.fft2(x)
    fr2 = np.fft.fft2(np.flipud(np.fliplr(y)))
    m, n = fr.shape
    cc = np.real(np.fft.ifft2(fr * fr2))
    cc = np.roll(cc, -int(m / 2) + 1, axis=0)
    cc = np.roll(cc, -int(n / 2) + 1, axis=1)
    return cc
    # <<< END


def _smooth(x: float, y: float, img: np.ndarray):
    gauss_kernel = Gaussian2DKernel(
        x_stddev=x, y_stddev=y, x_size=img.shape[0], y_size=img.shape[1]
    ).array
    return _convolve2d(img, gauss_kernel)


@dataclass
class CoordinateScale:
    linear: un.Quantity
    log: un.Quantity | None

    def get_scale(self, mode: str | None):
        match mode:
            case "linear" | None:
                return self.linear
            case "log":
                return self.log


class Grid:
    def __init__(
        self,
        model: DiskModel,
        theta_steps: int,
        r_scale: str | None = "log",
        theta_tol: float = 0.1,
    ):
        N_r, N_phi = model._run.get_polar_img_size()
        self.N_r: int = N_r
        self.N_phi: int = N_phi
        self.N_theta: int = theta_steps

        r_min, r_max = model.get_radius_lims()
        self.r_min: un.Quantity = (r_min * un.AU).to(un.meter)
        self.r_max: un.Quantity = (r_max * un.AU).to(un.meter)

        self._radii: CoordinateScale = CoordinateScale(
            linear=np.linspace(r_min.value, r_max.value, N_r) * un.meter,
            log=np.logspace(
                np.log10(r_min.value),
                np.log10(
                    r_max.value,
                ),
                N_r,
            )
            * un.meter,
        )
        self.radii = un.Quantity = self._radii.get_scale(mode=r_scale)

        self._r_edges: CoordinateScale = CoordinateScale(
            linear=np.linspace(r_min.value, r_max.value, N_r + 1, dtype=np.float64)
            * un.meter,
            log=np.logspace(
                np.log10(r_min.value),
                np.log10(
                    r_max.value,
                ),
                N_r + 1,
                dtype=np.float64,
            )
            * un.meter,
        )
        self.r_edges: un.Quantity = self._r_edges.get_scale(mode=r_scale)

        self._phis: CoordinateScale = CoordinateScale(
            linear=np.linspace(0, 2 * np.pi, N_phi) * un.radian,
            log=None,
        )
        self.phis: un.Quantity = self._phis.get_scale(
            mode="linear"
        )  # For now, only linear phi allowed
        self._phi_edges: CoordinateScale = CoordinateScale(
            linear=np.linspace(0, 2 * np.pi, N_phi + 1) * un.radian,
            log=None,
        )
        self.phi_edges: un.Quantity = self._phi_edges(mode="linear")

        self.heights: un.Quantity = model.get_height(radius=self.radii).to(un.meter)

        theta_max = np.abs(
            np.arccos(self.heights[-1] / self.radii[-1]).value - np.pi / 2
        )
        theta_tol = 0.1

        self._thetas: CoordinateScale = CoordinateScale(
            linear=np.linspace(
                np.pi / 2 - theta_max * (1 + theta_tol),
                np.pi / 2 + theta_max * (1 + theta_tol),
                self.N_theta,
            )
            * un.radian,
            log=None,
        )
        self.thetas: un.Quantity = self._thetas.get_scale(
            mode="linear"
        )  # For now, only linear theta allowed
        self._theta_edges: CoordinateScale = CoordinateScale(
            linear=np.linspace(
                np.pi / 2 - theta_max * (1 + theta_tol),
                np.pi / 2 + theta_max * (1 + theta_tol),
                self.N_theta + 1,
            )
            * un.radian,
            log=None,
        )
        self.theta_edges: un.Quantity = self._theta_edges(mode="linear")

    def get_polar_grid(self, mode: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        radii = self.radii if mode is None else self._radii.get_scale(mode=mode)
        phis = self.phis if mode is None else self._phis.get_scale(mode=mode)
        return np.meshgrid(radii.value, phis.value)


class RADMCSetup:
    def __init__(
        self,
        model: DiskModel,
        nphot_therm: int,
        nphot_scat: int,
        theta_steps: int,
        num_threads: int,
        r_scale: str | None = "log",
        theta_tol: float = 0.1,
        fast_mode: int = 0,
        modified_random_walk: bool = True,
        scattering_mode: int = 1,
    ):
        max_seed = np.iinfo(np.int32).max

        self.model: DiskModel = model
        self.nphot_therm: int = nphot_therm
        self.nphot_scat: int = nphot_scat
        self.num_threads: int = num_threads
        self.fast_mode: int = fast_mode
        self.modified_random_walk: bool = modified_random_walk
        self.scattering_mode: int = scattering_mode
        self.seed: int = self.model.get_rng().integers(
            low=-(max_seed - 1), high=max_seed - 1, dtype=np.int32
        )

        self.grid: Grid = Grid(
            model=model,
            theta_steps=theta_steps,
            r_scale=r_scale,
            theta_tol=theta_tol,
        )

        self.get_file_directory().mkdir(exist_ok=True, parents=True)

    def get_file_directory(self) -> Path:
        return self.model._directory / "radmc3d"

    def save_input_file(self, name: str, data: list, suffix="inp") -> None:
        with open(self.get_file_directory() / f"{name}.inp", "w") as file:
            file.writelines(format_output_lines(inp=data))

    def create_radmc3d_input(self) -> None:
        settings = {
            "nphot_therm": self.nphot_therm,
            "nphot_scat": self.nphot_scat,
            "nphot_spec": 10_000,
            "nphot_mono": 100_000,
            "iseed": self.seed,
            "ifast": int(self.fast_mode),
            "setthreads": self.num_threads,
            "modified_random_walk": int(self.modified_random_walk),
            "scattering_mode_max": self.scattering_mode,
        }

        self.save_input_file(
            name="radmc3d",
            data=list(map(lambda item: f"{item[0]}={item[1]}", settings.items())),
        )

    def create_amr_grid_input(self) -> None:
        grid_output = [
            "1",  # iformat
            "0",  # gridstyle (regular)
            "150",  # coordinate system (100 <= 150 < 200) -> spherical
            "0",  # gridinfo (no redundant information)
            "1 1 1",  # incl_x=1 incl_y=1 incl_z=1
        ]

        grid_output.append(
            f"{self.grid.N_r} {self.grid.N_theta} {self.grid.N_phi}"
        )  # nx ny nz
        grid_output.append(
            " ".join([str(i) for i in self.grid.r_edges.to(un.centimeter).value])
        )
        grid_output.append(
            " ".join([str(i) for i in self.grid.theta_edges.to(un.radian).value])
        )
        grid_output.append(
            " ".join([str(i) for i in self.grid.phi_edges.to(un.radian).value])
        )

        self.save_input_file(name="amr_grid", data=grid_output)
