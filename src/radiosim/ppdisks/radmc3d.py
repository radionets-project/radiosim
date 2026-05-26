from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import astropy.units as un
import numpy as np
from astropy.constants import M_sun, R_sun, c
from astropy.convolution import Gaussian2DKernel
from tqdm.auto import tqdm

from .config import Variables
from .config.radmc3d import format_output_lines

if TYPE_CHECKING:
    from radiosim.ppdisks.simulation import DiskModel


# FFT Convolution algorithm from https://stackoverflow.com/a/47979802
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
        theta_scale: str | None = "log",
        theta_log_exp: float = -3,
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
            linear=np.linspace(self.r_min.value, self.r_max.value, N_r) * un.meter,
            log=np.logspace(
                np.log10(self.r_min.value),
                np.log10(
                    self.r_max.value,
                ),
                N_r,
            )
            * un.meter,
        )
        self.radii = un.Quantity = self._radii.get_scale(mode=r_scale)

        self._r_edges: CoordinateScale = CoordinateScale(
            linear=np.linspace(
                self.r_min.value, self.r_max.value, N_r + 1, dtype=np.float64
            )
            * un.meter,
            log=np.logspace(
                np.log10(self.r_min.value),
                np.log10(
                    self.r_max.value,
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
        self.phi_edges: un.Quantity = self._phi_edges.get_scale(mode="linear")

        self.heights: un.Quantity = model.get_height(radius=self.radii).to(un.meter)

        if self.N_theta > 1:
            symlog = (
                np.concatenate(
                    [
                        -np.logspace(theta_log_exp, 1, self.N_theta // 2 - 1)[::-1],
                        np.zeros(1),
                        np.logspace(theta_log_exp, 1, self.N_theta // 2),
                    ]
                )
                / 10
            )
            symlog_edges = (
                np.concatenate(
                    [
                        -np.logspace(theta_log_exp, 1, self.N_theta // 2)[::-1],
                        np.zeros(1),
                        np.logspace(theta_log_exp, 1, self.N_theta // 2),
                    ]
                )
                / 10
            )

        theta_max = np.abs(
            np.arccos(self.heights[-1] / self.radii[-1]).value - np.pi / 2
        )

        self._thetas: CoordinateScale = CoordinateScale(
            linear=np.linspace(
                np.pi / 2 - theta_max * (1 + theta_tol),
                np.pi / 2 + theta_max * (1 + theta_tol),
                self.N_theta,
            )
            * un.radian,
            log=(symlog + np.pi / 2) * un.radian if self.N_theta > 1 else None,
        )
        self.thetas: un.Quantity = self._thetas.get_scale(mode=theta_scale)
        self._theta_edges: CoordinateScale = CoordinateScale(
            linear=np.linspace(
                np.pi / 2 - theta_max * (1 + theta_tol),
                np.pi / 2 + theta_max * (1 + theta_tol),
                self.N_theta + 1,
            )
            * un.radian,
            log=(symlog_edges + np.pi / 2) * un.radian if self.N_theta > 1 else None,
        )
        self.theta_edges: un.Quantity = self._theta_edges.get_scale(mode="linear")

    def get_polar_grid(
        self, r_mode: str | None = None, phi_mode: str | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        radii = self.radii if r_mode is None else self._radii.get_scale(mode=r_mode)
        phis = self.phis if phi_mode is None else self._phis.get_scale(mode=phi_mode)
        return np.meshgrid(radii.value, phis.value)


class RADMCSetup:
    def __init__(
        self,
        model: DiskModel,
        ref_frequency: float | un.Quantity,
        frequency_res: int,
        nphot_therm: int,
        nphot_scat: int,
        theta_steps: int,
        num_threads: int,
        r_scale: str | None = "log",
        theta_scale: str | None = "log",
        theta_log_exp: float = -3.0,
        theta_tol: float = 0.1,
        fast_mode: int = 0,
        modified_random_walk: bool = True,
        scattering_mode: int = 1,
    ):
        max_seed = np.iinfo(np.int32).max

        self.model: DiskModel = model
        self.ref_frequency: un.Quantity = (
            ref_frequency * un.hertz
            if isinstance(ref_frequency, float)
            else ref_frequency
        )
        self.ref_wavelength: un.Quantity = ((c / ref_frequency).decompose()).to(
            un.micrometer
        )
        self.frequency_res: int = frequency_res
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
            theta_scale=theta_scale,
            theta_log_exp=theta_log_exp,
            theta_tol=theta_tol,
        )

        self.get_file_directory().mkdir(exist_ok=True, parents=True)

    def get_file_directory(self) -> Path:
        return self.model.get_radmc3d_directory()

    def get_image_directory(self) -> Path:
        return self.model.get_image_directory()

    def get_command_prefix(self) -> str:
        return Variables.get("RADMC3D_ROOT") / "src/radmc3d"

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

    def create_dust_density_input(self) -> None:
        dust_density_output = [
            "1",  # iformat
            str(self.grid.N_r * self.grid.N_theta * self.grid.N_phi),  # nrcells
            str(self.model.get_num_species()),  # nspec (# of dust species)
        ]

        unit_system = self.model._run._sim._unit_system

        density_unit = 1 * unit_system.mass / unit_system.length**3

        for ispec in np.arange(1, self.model.get_num_species() + 1):
            data = (
                self.model.get_dust_density_3d(
                    output_idx=-1, dust_idx=ispec, r_scale=None, grid=self.grid
                )
                * density_unit
            ).cgs.value.ravel(order="F")
            dust_density_output.extend(data.tolist())

        self.save_input_file(name="dust_density", data=dust_density_output)

    def _get_wavelenghts(self) -> list[float]:
        star_temps = self.model.get_sample_config()[
            "planet_parameters.stellar_temperature"
        ]
        return np.logspace(
            -0.95 if np.max(star_temps) <= 10000 else -1.5,
            np.log10(self.ref_wavelength.to(un.micrometer).value),
            self.frequency_res,
        ).tolist()

    def create_wavelength_micron_input(self) -> None:
        output = [self.frequency_res]
        output.extend(self._get_wavelenghts())

        self.save_input_file(name="wavelength_micron", data=output)

    def create_stars_input(self) -> None:
        unit_system = self.model._run._sim._unit_system
        sample_config = self.model.get_sample_config()

        num_stars = (
            2 if sample_config["planet_parameters.binary_system"] else 1
        )  # nstars
        output = [
            "2",  # iformat
            f"{num_stars} {self.frequency_res}",  # nstars (# stars) nlam (# wavelength)
        ]
        # rstar[1] mstar[1] xstar[1] ystar[1] zstar[1]

        if num_stars == 2:
            for star_idx in range(num_stars):
                star_data = np.genfromtxt(
                    self.model.get_data_directory() / f"planet{star_idx}.dat"
                )

                r_star = (star_data[1] * unit_system.length).cgs().value
                phi_star = star_data[2]

                R_star = 1
                m_star = (
                    np.array(sample_config["planet_parameters.stellar_mass"]) * M_sun
                ).cgs.value[star_idx]
                x_star = r_star * np.cos(phi_star)
                y_star = r_star * np.sin(phi_star)
        else:
            R_star = R_sun.to(un.centimeter).value
            m_star = (
                np.array(sample_config["planet_parameters.stellar_mass"]) * M_sun
            ).cgs.value[0]
            x_star = 0
            y_star = 0

        output.append(f"{R_star} {m_star} {x_star} {y_star} {0}")
        output.extend(self._get_wavelenghts())
        output.extend(
            (-np.array(sample_config["planet_parameters.stellar_temperature"])).tolist()
        )  # black body temperatures -> negative sign

        self.save_input_file(name="stars", data=output)

    def create_camera_wavelength_micron_input(self) -> None:
        output = ["1", str(self.ref_wavelength.to(un.micrometer).value)]
        self.save_input_file(name="camera_wavelength_micron", data=output)

    def create_dustopac_input(self) -> None:
        output = [
            2,
            self.model.get_num_species(),
            "-----------------------------",
        ]

        for ispec in range(1, self.model.get_num_species() + 1):
            output.extend([1, 0, f"dust{ispec}"])

        self.save_input_file(name="dustopac", data=output)

    def create_dustkappa_input(self) -> None:
        wavelengths = self._get_wavelenghts()

        for ispec in range(1, self.model.get_num_species() + 1):
            opac = self.model.get_opacities(
                dust_idx=ispec, wavelengths=np.array(wavelengths)
            )

            output = [
                3,  # lambda kappa_abs kappa_scat g
                self.frequency_res,  # nlambda
            ]
            for i in range(0, len(wavelengths)):
                output.append(
                    f"{wavelengths[i]} {opac['k_abs'][0, i]} "
                    f"{opac['k_sca'][0, i]} {opac['g'][0, i]}"
                )

            self.save_input_file(name=f"dustkappa_{'dust'}{ispec}", data=output)

    # Subprocess output capture adapted from https://stackoverflow.com/a/28319191
    # Marked code (inside >>> BEGIN / <<< END) is licensed under CC BY-SA 3.0
    def run_mctherm(
        self,
        show_progress: bool = True,
        return_execution_time: bool = True,
        verbose: bool = False,
    ) -> dict | None:
        pass
        total_steps = self.nphot_therm
        model_desc = f" | Model {self.model._id}"

        cmd = f"{self.get_command_prefix()} mctherm"

        if verbose:
            print(f"CMD @ {self.get_file_directory()}  $ {cmd}")

        starting_time = time.time_ns()
        execution_times = {
            "mctherm_output_times": [],
            "mctherm_runtime": 0,
        }

        # >>> BEGIN
        with (
            tqdm(
                "Running Monte Carlo" + model_desc,
                total=total_steps,
                disable=not show_progress,
            ) as progress,
            subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=None if verbose else subprocess.DEVNULL,
                bufsize=1,
                universal_newlines=True,
                shell=True,
                cwd=self.get_file_directory(),
            ) as p,
        ):
            for line in p.stdout:
                if "photon nr:" not in line.lower():
                    continue

                execution_times["mctherm_output_times"].append(
                    time.time_ns() - starting_time
                )
                progress.n = int(line.lower().split("photon nr:")[-1].strip())
                progress.refresh()
        # <<< END

        execution_times["mctherm_runtime"] = time.time_ns() - starting_time

        if return_execution_time:
            return execution_times
        else:
            return None

    # Subprocess output capture adapted from https://stackoverflow.com/a/28319191
    # Marked code (inside >>> BEGIN / <<< END) is licensed under CC BY-SA 3.0
    def run_image(
        self,
        incl: float,
        phi: float,
        posang: float,
        image_idx: int | None = None,
        preserve_output_files: bool = False,
        show_progress: bool = True,
        return_execution_time: bool = True,
        verbose: bool = False,
    ) -> dict | None:
        total_steps = self.nphot_scat
        model_desc = f" | Model {self.model._id}"

        x_pix, y_pix = self.model._run._sim._output_img_size

        cmd = (
            f"{self.get_command_prefix()} image "
            f"lambda {self.ref_wavelength.value} "
            f"incl {incl} phi {phi} posang {posang} "
            f"npixx {x_pix} npixy {y_pix}"
        )

        if verbose:
            print(f"CMD @ {self.get_file_directory()}  $ {cmd}")

        starting_time = time.time_ns()
        execution_times = {
            "image_output_times": [],
            "image_runtime": 0,
        }

        # >>> BEGIN
        if total_steps > 0:
            with (
                tqdm(
                    "Scattering Monte Carlo" + model_desc,
                    total=total_steps,
                    disable=not show_progress,
                ) as progress,
                subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=None if verbose else subprocess.DEVNULL,
                    bufsize=1,
                    universal_newlines=True,
                    shell=True,
                    cwd=self.get_file_directory(),
                ) as p,
            ):
                for line in p.stdout:
                    if "photon nr:" not in line.lower():
                        continue

                    execution_times["image_output_times"].append(
                        time.time_ns() - starting_time
                    )
                    progress.n = int(line.lower().split("photon nr:")[-1].strip())
                    progress.refresh()
            # <<< END
        else:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL if not verbose else None,
                stderr=subprocess.DEVNULL if not verbose else None,
                shell=True,
                cwd=self.get_file_directory(),
            )

        execution_times["image_runtime"] = time.time_ns() - starting_time

        image_file = f"image_{image_idx}.out"
        self.get_image_directory().mkdir(exist_ok=True, parents=True)

        if preserve_output_files:
            shutil.copy(
                self.get_file_directory() / "image.out",
                self.get_image_directory() / image_file,
            )
        else:
            shutil.move(
                self.get_file_directory() / "image.out",
                self.get_image_directory() / image_file,
            )

        if return_execution_time:
            return execution_times
        else:
            return None
