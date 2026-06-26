import pickle
import shutil
import warnings
from os import PathLike
from pathlib import Path
from time import time

import dsharp_opac as opacities
import matplotlib
import matplotlib.animation as animation
import numpy as np
from astropy import constants as const
from astropy import units as un
from astropy.time import Time
from numpy.typing import ArrayLike
from scipy import integrate
from scipy.interpolate import PchipInterpolator
from tqdm.auto import tqdm

from radiosim.ppdisks.config import TOMLConfiguration
from radiosim.ppdisks.plotting.plotting import plot_image
from radiosim.ppdisks.plotting.utils import ellipse_img2cartesian_img

from .config import Variables
from .config.fargo import Constants, Planet, PlanetConfig, UnitSystem
from .disk_functions import (
    approximate_grain_size,
    diffusion_coefficient,
    disk_height,
    mass_function,
    orbital_period,
    sigma0,
    surface_density,
)
from .plotting.utils import configure_axes
from .radmc3d import Grid, RADMCSetup
from .setup import Setup

__all__ = ["Simulation", "SimulationRun", "DiskModel"]


def get_default_sampling_config():
    return {
        "disk_parameters": {
            "aspect_ratio": [0.01, 0.1],  # Disk aspect ratio @ r=R0 (default R0 = 1AU)
            "disk_mass_ref_radius": 150,  # Reference radius R_ref in AU
            "disk_mass": [0.01, 0.03],  # Cumulative disk mask in M_sun @ r=R_ref
            "sigma_slope": [0.1, 0.3],  # Exponent of the density profile
            "flaring_index": [0.5, 2.0],
            "alpha": [0.001, 0.01],  # Shakura-Sunyaev viscosity parameter
        },
        "dust_parameters": {
            "invstokes": {
                "1": [10.0, 20.0],
            },
            "epsilon": [0.05, 0.2],  # dust-to-gas ratio,
        },
        "planet_parameters": {
            "binary_ratio": 0.0,  # Ratio of binary systems to single systems
            "binary_period": [6.04800e5, 3e7],  # Seconds (logarithmic sampling)
            "binary_eccentricity": [0.0, 0.2],  # 0 = Circle, 0 < e < 1 = Ellipse
            "stellar_mass": [0.5, 2],  # Solar Masses
            "stellar_temperature": [3000.0, 6000.0],  # Kelvin
            "num_planets": [2, 3],
            "planet_mass": [1.0e-6, 5.0e-3],  # Solar Masses
            "planet_orbit_radius": [6.0, 15.0],  # Astronomical Units
            # short: PEF -> no other planets allowed closer than R_orbit * PEF
            "planet_exclusion_factor": 0.2,
            "planet_exclusion_max_iter": 100,  # max iterations to determine valid orbit
            "eccentricity": [0.0, 0.1],  # 0 = Circle, 0 < e < 1 = Ellipse
        },
        "mesh_parameters": {
            "y_min": [3.0, 5.0],  # Astronomical Units
            "y_max_ratio": [1.5, 3],  # Multiple of max(orbital_radius)
        },
        "extrapolation_parameters": {
            "extrapolation_active": True,  # whether to extrapolate the dustdens
            "extrapolation_cutoff_idx": 30,  # dustdens values included in extrapolation
            "density_rim_extend_factor": [
                0.7,
                3.0,
            ],  # multiple of value @ smallest radius
            "r_rim_maxium_factor": [
                0.3,
                0.7,
            ],  # factor of position of inner rim maximum
            "r_min": [0.1, 1.0],  # minimal radius in AU
        },
        "output_parameters": {
            "num_largest_orbits": [100, 200],
        },
        "grid_parameters": {
            "r_scale": "log",
            "theta_scale": "log",
            "theta_steps": 500,
            "theta_log_exp": -1.5,
            "theta_tol": 0.1,
        },
        "thermal_mc_parameters": {
            "scattering_mode": 1,  # Scattering mode:
            # 0: no scattering
            # 1: isotropic scattering
            # 2: anisotropic scattering
            # 3 - 5 --> see radmc3d manual
            "fast_mode": 0,  # Whether to use 'fast mode'
            "modified_random_walk": True,  # Whether to use MRW
            "freq_res": 1000,  # num of frequencies tested for the MC run
            "nphot_therm": 1_000_000_000,  # num of thermal photon packages for MC run
        },
        "imaging_parameters": {
            "nphot_scat": 0,  # num of scattering photon packages for the imaging run
            "num_versions": [1, 2],  # max uses of the same dust distribution
            "incl": [0.0, 30.0],  # inclination of the camera relative to image plane
            "phi": [0.0, 45.0],  # polar angle of the camera relative to image plane
            "posang": [
                0.0,
                45.0,
            ],  # position angle of the camera relative to image plane
        },
    }


class Simulation:
    def __init__(
        self,
        name: str,
        root_directory: PathLike,
        setup: Setup,
        float_type: type,
        polar_img_size: tuple[int],
        output_img_size: tuple[int],
        ref_freq: float | un.Quantity,
        unit_system: UnitSystem,
        use_default_constants: bool,
    ):
        self.name: str = name
        self._root_directory: Path = Path(root_directory)

        self._out_directory: Path = self._root_directory / "outputs"
        self._out_directory.mkdir(exist_ok=True, parents=True)

        self._setup: Setup = setup

        self._sampling_config: TOMLConfiguration = TOMLConfiguration(
            self._root_directory / "sampling_config.toml", create_if_not_exists=True
        )

        self._config: TOMLConfiguration = TOMLConfiguration(
            self._root_directory / "config.toml", create_if_not_exists=True
        )

        self._float_type: type = float_type

        self._ref_freq: un.Quantity = (
            ref_freq if isinstance(ref_freq, un.Quantity) else ref_freq * un.hertz
        )
        self._ref_wavelength: un.Quantity = ((const.c / self._ref_freq).decompose()).to(
            un.micrometer
        )

        self._polar_img_size: tuple[int] = polar_img_size
        self._output_img_size: tuple[int] = output_img_size

        self._unit_system: UnitSystem = unit_system
        self._use_default_constants: bool = use_default_constants

        if use_default_constants:
            self._constants: Constants = Constants.default(unit_system=unit_system)
            self._constants._autosave = True
            self._constants.save()
        else:
            self._constants: Constants = Constants(
                unit_system=unit_system, autosave=True
            )

        self._planet_config: PlanetConfig = PlanetConfig(
            name=f"radiosim_{self.name}", autosave=True, unit_system=self._unit_system
        )

    def save_config(self) -> None:
        content = {
            "general": {
                "name": self.name,
                "root_directory": str(self._root_directory.expanduser()),
                "setup": self._setup._name,
                "float_type": "FLOAT64"
                if self._float_type == np.float64
                else "FLOAT32",
                "ref_freq": self._ref_freq.to(un.hertz).value,
                "polar_img_size": list(self._polar_img_size),
                "output_img_size": list(self._output_img_size),
                "unit_system": self._unit_system.name,
                "use_default_constants": self._use_default_constants,
            }
        }
        self._config.dump_dict(content)

    def get_next_run_id(self) -> int:
        dirs = [d for d in self._out_directory.glob("run_*") if d.is_dir()]
        if len(dirs) == 0:
            return 0

        dir_ids = [int(str(d.name).removeprefix("run_")) for d in dirs]
        return np.max(dir_ids) + 1

    def get_runs(self) -> list["SimulationRun"]:
        dirs = [d for d in self._out_directory.glob("run_*") if d.is_dir()]
        if len(dirs) == 0:
            return []

        run_ids = [int(str(d.name).removeprefix("run_")) for d in dirs]
        return [SimulationRun(id=run_id, sim=self) for run_id in run_ids]

    def get_run(self, run_id: int) -> "SimulationRun":
        return SimulationRun(id=run_id, sim=self)

    def simulate(
        self,
        num_images: int,
        seed: int,
        num_outputs: int | None = None,
        steps_per_orbit: int | None = None,
        run_id: int | None = None,
        resume: bool = True,
        resume_model_id: int | None = None,
        gpu: bool = True,
        cuda_device_id: int = 0,
        parallel: bool = False,
        num_nodes: int = 1,
        num_mc_threads: int = 1,
        override_samples: dict | None = None,
        force_manual_mode: bool = False,
        show_progress: bool = True,
        record_execution_time: bool = True,
        verbose: bool = False,
        overwrite: bool = False,
    ) -> None:
        manual_run = False

        if override_samples is not None:
            if not force_manual_mode:
                confirmation = input(
                    "Are you sure you want to use a manual sampling? (y/N)"
                )

                if confirmation.lower() == "y":
                    manual_run = True
                else:
                    print("Manual run canceled. Aborting.")
                    return None
            else:
                print("Forcing manual run.")
                manual_run = True

        if run_id is None:
            run = SimulationRun.new(
                num_images=num_images,
                steps_per_orbit=steps_per_orbit,
                num_outputs=num_outputs,
                seed=seed,
                sim=self,
            )
            resume = False
        else:
            run = SimulationRun(id=run_id, sim=self, resume_rng=False)
            num_images = run.get_num_images()

        print(f"------ STARTING RUN {run._id} ------")
        if manual_run:
            print("----- ! MANUAL MODE ACTIVE ! -----")

        num_current_images = run.get_num_current_images()
        if not resume or num_current_images == 0:
            start_idx = 0
        elif resume and resume_model_id is not None:
            start_idx = (
                resume_model_id
                if resume_model_id > 0
                else run.get_models()[resume_model_id]._id
            )
        else:
            start_idx = num_current_images

        for i in np.arange(start_idx, num_images):
            skip_fargo = False
            skip_radmc = False

            model = DiskModel(id=i, run=run)
            existed_prev = True

            if not model.exists():
                model = DiskModel.new(id=i, run=run)
                existed_prev = False

            skip_fargo = (
                model.get_data_directory()
                / f"dust1dens{model.get_num_outputs() - 1}.dat"
            ).exists()

            skip_radmc = model.get_image_directory().exists() and bool(
                np.all(
                    [
                        (model.get_image_directory() / f"image_{j}.out").exists()
                        for j in range(0, model.get_num_images())
                    ]
                )
            )

            if existed_prev and verbose:
                print(f"Resuming at Model {model._id}")
                print(f"{skip_fargo=}")
                print(f"{skip_radmc=}")

            if not manual_run:
                samples = run.draw_samples(model_id=model._id)  # current new model id
                run.save_rng(model_id=model._id)  # current new model id
            else:
                samples = override_samples

            # Dump samples to TOML file

            def toml_serialize_dict(read_dict):
                write_dict = dict()
                for key, value in read_dict.items():
                    if isinstance(value, dict):
                        write_dict[key] = toml_serialize_dict(read_dict=value)
                    elif isinstance(value, np.ndarray):
                        write_dict[key] = list(value)
                    elif isinstance(value, np.int64):
                        write_dict[key] = int(value)
                    else:
                        write_dict[key] = value
                return write_dict

            sample_config = model.get_sample_config()
            sample_config.create()

            sample_dump = samples.copy()
            sample_config.dump_dict(content=toml_serialize_dict(read_dict=sample_dump))

            # Run FARGO3D Simulation
            if not skip_fargo:
                fargo_compile_time, fargo_runtime = self._simulate_fargo(
                    run=run,
                    model=model,
                    samples=samples,
                    gpu=gpu,
                    parallel=parallel,
                    show_progress=show_progress,
                    verbose=verbose,
                    return_execution_time=record_execution_time,
                    num_nodes=num_nodes,
                    cuda_device_id=cuda_device_id,
                )

            # Run RADMC3D Simulation

            if not skip_radmc:
                radmc_runtimes = self._simulate_radmc(
                    run=run,
                    model=model,
                    samples=samples,
                    show_progress=show_progress,
                    verbose=verbose,
                    return_execution_time=record_execution_time,
                    num_mc_threads=num_mc_threads,
                )

            num_current_images += model.get_num_images()

            if record_execution_time:
                record_toml = TOMLConfiguration(
                    path=model.get_data_directory() / "execution_time.toml",
                    create_if_not_exists=False,
                )

                if record_toml.is_valid():
                    prev_values = record_toml.as_dict()
                else:
                    record_toml.create()
                    prev_values = None

                record_toml.dump_dict(
                    {
                        "mode": {
                            "gpu": gpu,
                            "parallel": parallel,
                            "num_nodes": num_nodes,
                            "cuda_device_id": cuda_device_id,
                        }
                        if prev_values is None
                        else prev_values["mode"],
                        "fargo_compile_time": fargo_compile_time
                        if prev_values is None
                        else prev_values["fargo_compile_time"],
                        "fargo_run_time": fargo_runtime[0]
                        if prev_values is None
                        else prev_values["fargo_run_time"],
                        "fargo_output_times": fargo_runtime[1]
                        if prev_values is None
                        else prev_values["fargo_output_times"],
                    }
                    | radmc_runtimes
                )

    def _simulate_fargo(
        self,
        run: "SimulationRun",
        model: "DiskModel",
        samples: dict,
        **kwargs,
    ) -> tuple[int, tuple[int]]:
        option_config = self._setup._option_config
        option_config._autosave = False
        param_config = self._setup._param_config
        param_config._autosave = False

        # Update Planet Config

        self._planet_config.clear()
        self._planet_config._autosave = False

        planet_parameters = samples["planet_parameters"]

        # See https://fargo3d.github.io/documentation/nbody.html
        if planet_parameters["binary_system"]:
            # First star: distance -> binary period
            self._planet_config.add_planet(
                planet=Planet(
                    name="star1",
                    distance=planet_parameters["binary_period"],
                    mass=planet_parameters["stellar_mass"][0] * const.M_sun,
                    feels_disk=False,
                    feels_others=True,
                    unit_system=self._unit_system,
                )
            )
            # Second star: distance -> binary eccentricity
            self._planet_config.add_planet(
                planet=Planet(
                    name="star2",
                    distance=planet_parameters["binary_eccentricity"],
                    mass=planet_parameters["stellar_mass"][1] * const.M_sun,
                    feels_disk=False,
                    feels_others=True,
                    unit_system=self._unit_system,
                )
            )

            m_star = np.sum(planet_parameters["stellar_mass"]) * const.M_sun
            self._constants["MSTAR"] = m_star
            option_config["planetary_system.NODEFAULTSTAR"].enable()
        else:
            m_star = planet_parameters["stellar_mass"][0] * const.M_sun
            self._constants["MSTAR"] = m_star
            option_config["planetary_system.NODEFAULTSTAR"].disable()

        for planet_idx in np.arange(0, planet_parameters["num_planets"]):
            self._planet_config.add_planet(
                Planet(
                    name=f"planet{planet_idx + 1}",
                    distance=planet_parameters["planet_orbit_radius"][planet_idx]
                    * un.AU,
                    mass=planet_parameters["planet_mass"][planet_idx] * const.M_sun,
                    feels_disk=True,
                    feels_others=True,
                    unit_system=self._unit_system,
                )
            )

        param_config["planet_parameters.planetConfig"] = "/".join(
            self._planet_config._path.parts[-2:]
        )
        param_config["planet_parameters.eccentricity"] = planet_parameters[
            "eccentricity"
        ]

        # Update Parameter Config
        ## Disk Parameters
        disk_parameters = samples["disk_parameters"]

        param_config["disk_parameters.aspectRatio"] = disk_parameters["aspect_ratio"]

        param_config["disk_parameters.sigma0"] = sigma0(
            ref_radius=(disk_parameters["disk_mass_ref_radius"] * un.AU)
            .to(self._unit_system.length)
            .value,
            R0=self._constants["R0"].value,
            mass=(disk_parameters["disk_mass"] * const.M_sun)
            .to(self._unit_system.mass)
            .value,
            sigma_slope=disk_parameters["sigma_slope"],
        )

        param_config["disk_parameters.sigmaSlope"] = disk_parameters["sigma_slope"]
        param_config["disk_parameters.flaringIndex"] = disk_parameters["flaring_index"]
        param_config["disk_parameters.alpha"] = disk_parameters["alpha"]

        ## Dust Parameters
        dust_parameters = samples["dust_parameters"]

        param_config["dust_parameters.epsilon"] = dust_parameters["epsilon"]

        for dust_idx, invstokes in dust_parameters["invstokes"].items():
            param_config[f"dust_parameters.invstokes{dust_idx}"] = invstokes

        ## Mesh Parameters
        mesh_parameters = samples["mesh_parameters"]

        distances = self._planet_config.get_distances()

        # If binary: distance -> binary eccentricity || binary period != distance
        # for the two stars in the planet file
        if planet_parameters["binary_system"]:
            distances = distances[2:]

        param_config["mesh_parameters.ymin"] = (
            (np.min([mesh_parameters["y_min"], distances.min()]) * un.AU)
            .to(self._unit_system.length)
            .value
        )

        max_orbit_radius = distances.max()
        param_config["mesh_parameters.ymax"] = (
            mesh_parameters["y_max_ratio"] * max_orbit_radius
        )

        param_config["mesh_parameters.nx"] = run.get_polar_img_size()[1]
        param_config["mesh_parameters.ny"] = run.get_polar_img_size()[0]

        ## Output Parameters

        output_parameters = samples["output_parameters"]

        num_orbits = output_parameters["num_largest_orbits"]

        def orbital_period(mass, radius, G):
            return np.sqrt((4 * np.pi**2 * radius**3) / (mass * G))

        period = orbital_period(
            mass=m_star,
            radius=max_orbit_radius * self._unit_system.length,
            G=self._constants["G"],
        )

        total_time = num_orbits * period
        step_size = period / run.get_steps_per_orbit()

        N_tot = int(total_time / step_size)
        N_interm = int(N_tot / run.get_num_outputs())

        param_config["output_parameters.dt"] = step_size.to(
            self._unit_system.time
        ).value
        param_config["output_parameters.ninterm"] = N_interm
        param_config["output_parameters.ntot"] = N_tot

        # Set output to fargo output directory
        # (to avoid overflow of OUTPUTDIR variable in C)
        param_config["output_parameters.outputDir"] = str(model.get_fargo_output_path())

        # Additional Parameters

        if run.get_float_type() == np.float64:
            option_config["performance.FLOAT"].disable()
        else:
            option_config["performance.FLOAT"].enable()

        # Save configurations

        self._planet_config.save()
        self._planet_config._autosave = True

        param_config.save()
        param_config._autosave = True

        option_config.save()
        option_config._autosave = True

        # Recompile and Run Setup

        compile_time = self._setup.compile(
            gpu=kwargs["gpu"],
            parallel=kwargs["parallel"],
            unit_system=self._unit_system,
            rescale=False,
            model_id=model._id,
            show_progress=kwargs["show_progress"],
            verbose=kwargs["verbose"],
            # show_fargo_output=kwargs["verbose"],
            show_fargo_output=False,
            return_execution_time=kwargs["return_execution_time"],
        )

        run_time = self._setup.run(
            model_id=model._id,
            num_nodes=kwargs["num_nodes"],
            parallel=kwargs["parallel"],
            show_progress=kwargs["show_progress"],
            cuda_device_id=kwargs["cuda_device_id"],
            verbose=kwargs["verbose"],
            return_execution_time=kwargs["return_execution_time"],
        )

        # Move the data files to the correct directory
        model.get_data_directory().mkdir()
        for file in (Variables.get("FARGO_ROOT") / model.get_fargo_output_path()).glob(
            "*.*"
        ):
            shutil.move(
                src=file,
                dst=model.get_data_directory(),
            )
        shutil.rmtree(path=Variables.get("FARGO_ROOT") / model.get_fargo_output_path())

        return compile_time, run_time

    def _simulate_radmc(
        self,
        run: "SimulationRun",
        model: "DiskModel",
        samples: dict,
        num_mc_threads: int,
        **kwargs,
    ) -> None:
        radmc_setup = RADMCSetup(
            model=model,
            ref_frequency=run._sim._ref_freq,
            frequency_res=samples["thermal_mc_parameters"]["freq_res"],
            nphot_therm=samples["thermal_mc_parameters"]["nphot_therm"],
            nphot_scat=samples["imaging_parameters"]["nphot_scat"],
            num_threads=num_mc_threads,
            fast_mode=samples["thermal_mc_parameters"]["fast_mode"],
            modified_random_walk=samples["thermal_mc_parameters"][
                "modified_random_walk"
            ],
            scattering_mode=samples["thermal_mc_parameters"]["scattering_mode"],
        )

        # Create input files
        radmc_setup.create_radmc3d_input()
        radmc_setup.create_amr_grid_input()
        radmc_setup.create_dust_density_input()
        radmc_setup.create_wavelength_micron_input()
        radmc_setup.create_stars_input()
        radmc_setup.create_camera_wavelength_micron_input()
        radmc_setup.create_dustopac_input()
        radmc_setup.create_dustkappa_input()

        runtimes = {}

        mctherm_runtime = radmc_setup.run_mctherm(
            show_progress=kwargs["show_progress"],
            return_execution_time=kwargs["return_execution_time"],
            verbose=kwargs["verbose"],
        )

        runtimes = runtimes | mctherm_runtime

        for i in range(0, model.get_num_images()):
            image_runtime = radmc_setup.run_image(
                image_idx=i,
                incl=samples["imaging_parameters"]["incl"][i],
                phi=samples["imaging_parameters"]["phi"][i],
                posang=samples["imaging_parameters"]["posang"][i],
                return_execution_time=kwargs["return_execution_time"],
                show_progress=kwargs["show_progress"],
                verbose=kwargs["verbose"],
            )
            runtimes = runtimes | {str(i): image_runtime}

        return runtimes

    @classmethod
    def new(
        cls,
        name: str,
        setup: str,
        sampling_config: PathLike | dict | None,
        ref_freq: float | un.Quantity,
        parent_directory: PathLike | None = None,
        float_type: type = np.float64,
        polar_img_size: tuple[int] = (300, 800),
        output_img_size: tuple[int] = (300, 300),
        unit_system: UnitSystem | str = UnitSystem.MKS,
        use_default_constants: bool = True,
    ) -> "Simulation":
        if parent_directory is None:
            parent_directory = Path.cwd()

        if float_type not in [np.float64, np.float32]:
            raise TypeError(
                "Only numpy.float64 or numpy.float32 are allowed floating point types."
            )

        root_directory = Path(parent_directory) / name

        if not root_directory.exists():
            root_directory.mkdir(parents=True, exist_ok=True)
        elif root_directory.is_dir():
            raise IsADirectoryError("This simulation already exists.")
        elif root_directory.is_file():
            raise TypeError("The root directory must be a directory but is a file!")

        if isinstance(sampling_config, dict):
            sampling_dict = sampling_config
            sampling_config = TOMLConfiguration(
                path=root_directory / "sampling_config.toml", create_if_not_exists=True
            )
            sampling_config.dump_dict(content=sampling_dict)
        elif sampling_config is None:
            sampling_config = TOMLConfiguration(
                path=root_directory / "sampling_config.toml", create_if_not_exists=True
            )
            sampling_config.dump_dict(content=get_default_sampling_config())
        else:
            sampling_config_path = Path(root_directory / "sampling_config.toml")
            shutil.copy(sampling_config, sampling_config_path)
            sampling_config = TOMLConfiguration(path=sampling_config_path)

        print(f"Created local sampling config at location {sampling_config.get_path()}")

        setup = Setup(name=setup)

        instance = cls(
            name=name,
            root_directory=root_directory,
            setup=setup,
            float_type=float_type,
            output_img_size=output_img_size,
            polar_img_size=polar_img_size,
            unit_system=unit_system,
            use_default_constants=use_default_constants,
            ref_freq=ref_freq,
        )

        instance.save_config()

        return instance

    @classmethod
    def load(cls, root_directory: PathLike) -> "Simulation":
        root_directory = Path(root_directory)
        config = TOMLConfiguration(path=root_directory / "config.toml")

        if not config.is_valid():
            raise ValueError("There is no valid configuration file in this directory.")

        instance = Simulation(
            name=config["general.name"],
            root_directory=root_directory,
            setup=Setup(name=config["general.setup"]),
            float_type=np.float64
            if config["general.float_type"] == "FLOAT64"
            else np.float32,
            ref_freq=config["general.ref_freq"],
            polar_img_size=tuple(config["general.polar_img_size"]),
            output_img_size=tuple(config["general.output_img_size"]),
            unit_system=UnitSystem.__members__[config["general.unit_system"]],
            use_default_constants=config["general.use_default_constants"],
        )

        return instance


class SimulationRun:
    def __init__(
        self,
        id: int,
        sim: Simulation,
        num_images: int | None = None,
        seed: int | None = None,
        resume_rng: bool = True,
    ):
        self._id: int = id
        self._directory: Path = sim._out_directory / f"run_{self._id}"

        self._sampling_config: TOMLConfiguration = TOMLConfiguration(
            self._directory / "sampling_config.toml",
            create_if_not_exists=True,
        )
        self._sim: Simulation = sim

        if num_images is None:
            self._num_images: int = self.get_num_images()
        else:
            self._num_images: int = num_images

        if seed is None:
            self._rng: np.random.Generator = np.random.default_rng(seed=self.get_seed())
        else:
            self._rng: np.random.Generator = np.random.default_rng(seed=seed)

        if resume_rng:
            model_id = self.get_next_model_id() - 1

            if model_id < 0:
                model_id = None

            self._rng = self.get_rng(model_id=model_id)

    def plot_times(
        self,
        mode: str = "total",
        show_total_time: bool = True,
        save_to: str | PathLike | None = None,
        save_args: dict | None = None,
        time_unit: un.Unit = un.second,
        color: str = "maroon",
        plot_args: dict | None = None,
        fig: matplotlib.figure.Figure | None = None,
        fig_args: dict | None = None,
        ax: matplotlib.axes.Axes | None = None,
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        fig_args = {} if fig_args is None else fig_args
        plot_args = {} if plot_args is None else plot_args
        save_args = {"bbox_inches": "tight"} if save_args is None else save_args

        fig, ax = configure_axes(fig=fig, ax=ax, fig_args=fig_args)

        models = self.get_models()
        ylabel = ""
        match mode:
            case "run":
                times = np.array(
                    [model.get_execution_times()["run_time"] for model in models]
                )
                ylabel = "Runtime"
            case "compile":
                times = np.array(
                    [model.get_execution_times()["compile_time"] for model in models]
                )
                ylabel = "Compile Time"
            case "total":
                times = np.zeros(len(models))
                for i in range(len(models)):
                    execution_times = models[i].get_execution_times()
                    times[i] = (
                        execution_times["run_time"] + execution_times["compile_time"]
                    )
                ylabel = "Total Time"
            case _:
                raise ValueError("Valid modes are 'run', 'compile', 'total'.")

        times *= un.nanosecond
        times = times.to(time_unit)

        if show_total_time:
            plot_args["label"] = f"Total Time: {np.round(np.sum(times), 3)}"

        model_ids = [model._id for model in models]
        ax.scatter(model_ids, times.value, color=color, **plot_args)

        ax.set_xlabel("Model ID")
        ax.set_ylabel(f"{ylabel} / {time_unit.to_string(format='latex')}")

        if show_total_time:
            ax.legend()

        if save_to is not None:
            fig.savefig(save_to, **save_args)

        return fig, ax

    def get_polar_img_size(self) -> tuple[int]:
        return tuple(self._sampling_config["run.polar_img_size"])

    def get_num_outputs(self) -> int:
        return self._sampling_config["run.num_outputs"]

    def get_steps_per_orbit(self) -> int:
        return self._sampling_config["run.steps_per_orbit"]

    def get_float_type(self) -> type:
        return (
            np.float64
            if self._sampling_config["run.float_type"] == "FLOAT64"
            else np.float32
        )

    def get_num_images(self) -> int:
        return self._sampling_config["run.num_images"]

    def get_num_current_images(self) -> int:
        return (
            np.sum(
                [
                    model.get_num_images()
                    if model.get_image_directory().exists()
                    else 0
                    for model in self.get_models()
                ]
            )
            if len(self.get_models()) > 0
            else 0
        )

    def get_seed(self) -> int:
        return self._sampling_config["run.seed"]

    def get_models(self) -> list["DiskModel"]:
        return [
            DiskModel(id=int(str(d.name).removeprefix("model_")), run=self)
            for d in self._directory.glob("model_*")
            if d.is_dir()
        ]

    def get_model(self, id: int) -> "DiskModel":
        for model in self.get_models():
            if model._id == id:
                return model

        raise KeyError(f"There is no disk model with id '{id}' in run '{self._id}'.")

    def get_next_model_id(self) -> int:
        dirs = [d for d in self._directory.glob("model_*") if d.is_dir()]
        if len(dirs) == 0:
            return 0

        dir_ids = [int(str(d.name).removeprefix("model_")) for d in dirs]
        return np.max(dir_ids) + 1

    def get_rng(self, model_id: int | None = None) -> np.random.Generator:
        if model_id is None:
            return self._rng

        return self.get_model(id=model_id).get_rng()

    def save_rng(self, model_id: int | None = None) -> None:
        model = self.get_model(
            id=self.get_next_model_id() - 1 if model_id is None else model_id
        )
        rng_path = model._directory / "rng_state.pkl"

        with open(rng_path, "wb") as pkl:
            pickle.dump(self._rng, pkl, pickle.HIGHEST_PROTOCOL)

    def draw_samples(self, model_id: int | None = None) -> dict:
        sampling_config = self._sampling_config.as_dict()
        rng = self.get_rng(
            model_id=model_id - 1 if model_id > 0 else None
        )  # get RNG from previous model

        def sample_dict(read_dict):
            write_dict = dict()
            for key, value in read_dict.items():
                if isinstance(value, dict):
                    write_dict[key] = sample_dict(read_dict=value)
                elif isinstance(value, list):
                    if isinstance(value[0], int):
                        write_dict[key] = rng.integers(low=value[0], high=value[1])
                    else:
                        write_dict[key] = rng.uniform(low=value[0], high=value[1])
                elif np.isscalar(value) or isinstance(value, (str, bool)):
                    write_dict[key] = value

            return write_dict

        disk_parameters = sample_dict(sampling_config["disk_parameters"])
        dust_parameters = sample_dict(sampling_config["dust_parameters"])

        # Planetary system parameters
        planet_sampling = sampling_config["planet_parameters"]

        planet_parameters = {}

        planet_parameters["binary_system"] = (
            rng.uniform(0, 1) <= planet_sampling["binary_ratio"]
        )
        if planet_parameters["binary_system"]:
            num_stars = 2
            planet_parameters["binary_period"] = 10 ** rng.uniform(
                low=np.log10(planet_sampling["binary_period"][0]),
                high=np.log10(planet_sampling["binary_period"][1]),
            )
            planet_parameters["binary_eccentricity"] = rng.uniform(
                low=planet_sampling["binary_eccentricity"][0],
                high=planet_sampling["binary_eccentricity"][1],
            )
        else:
            num_stars = 1

        planet_parameters["stellar_mass"] = rng.uniform(
            low=planet_sampling["stellar_mass"][0],
            high=planet_sampling["stellar_mass"][1],
            size=num_stars,
        )

        planet_parameters["stellar_temperature"] = rng.uniform(
            low=planet_sampling["stellar_temperature"][0],
            high=planet_sampling["stellar_temperature"][1],
            size=num_stars,
        )

        num_planets = rng.integers(
            low=planet_sampling["num_planets"][0],
            high=planet_sampling["num_planets"][1],
            endpoint=True,
        )

        planet_parameters["num_planets"] = num_planets

        planet_parameters["planet_mass"] = rng.uniform(
            low=planet_sampling["planet_mass"][0],
            high=planet_sampling["planet_mass"][1],
            size=num_planets,
        )

        planet_orbits = np.zeros(num_planets)

        # This makes sure that planets are not too close to each other
        for i_planet in range(num_planets):
            valid = False

            max_iter = 100
            i = 0
            while not valid:
                orbit = rng.uniform(
                    low=planet_sampling["planet_orbit_radius"][0],
                    high=planet_sampling["planet_orbit_radius"][1],
                )

                # if new orbit violates planet exclusion zones around other planets
                if (
                    np.any(
                        np.abs(planet_orbits[:i_planet] - orbit)
                        / planet_orbits[:i_planet]
                        < planet_sampling["planet_exclusion_factor"]
                    )
                    and i < max_iter
                ):
                    i += 1
                    continue
                else:
                    valid = True

                if i >= max_iter:
                    warnings.warn(
                        f"The maximum iteration number ({max_iter}) for "
                        "determining a planet's orbit was exceeded! The generated orbit"
                        " thus violates the exclusion zone. Consider reducing the "
                        "number of planets, the exclusion factor or increase the "
                        "maximum radius.",
                        stacklevel=1,
                    )

            planet_orbits[i_planet] = orbit

        planet_parameters["planet_orbit_radius"] = planet_orbits

        planet_parameters["eccentricity"] = rng.uniform(
            low=planet_sampling["eccentricity"][0],
            high=planet_sampling["eccentricity"][1],
        )

        mesh_parameters = sample_dict(sampling_config["mesh_parameters"])

        if sampling_config["extrapolation_parameters"][
            "extrapolation_active"
        ] and np.max(sampling_config["extrapolation_parameters"]["r_min"]) >= np.min(
            sampling_config["mesh_parameters"]["y_min"]
        ):
            raise ValueError(
                "The minimum radius for the extrapolated inner part of the disk is "
                "greater or equal to the minimum possible radius of the FARGO mesh!"
            )

        if (
            np.max(sampling_config["extrapolation_parameters"]["r_rim_maxium_factor"])
            > 1
        ):
            raise ValueError(
                "The position of the maximum of the extrapolated inner ring may not be "
                "outside the rim area! Thus the extension factor must be smaller 1!"
            )

        extrapolation_parameters = sample_dict(
            sampling_config["extrapolation_parameters"]
        )

        output_parameters = sample_dict(sampling_config["output_parameters"])
        grid_parameters = sample_dict(sampling_config["grid_parameters"])
        thermal_mc_parameters = sample_dict(sampling_config["thermal_mc_parameters"])

        imaging_sampling = sampling_config["imaging_parameters"]
        imaging_parameters = {
            "nphot_scat": imaging_sampling["nphot_scat"],
            "num_versions": rng.integers(
                low=imaging_sampling["num_versions"][0],
                high=imaging_sampling["num_versions"][1],
                endpoint=True,
            ),
        }

        imaging_parameters["incl"] = rng.uniform(
            low=imaging_sampling["incl"][0],
            high=imaging_sampling["incl"][1],
            size=int(imaging_parameters["num_versions"]),
        )

        imaging_parameters["phi"] = rng.uniform(
            low=imaging_sampling["phi"][0],
            high=imaging_sampling["phi"][1],
            size=int(imaging_parameters["num_versions"]),
        )

        imaging_parameters["posang"] = rng.uniform(
            low=imaging_sampling["posang"][0],
            high=imaging_sampling["posang"][1],
            size=int(imaging_parameters["num_versions"]),
        )

        samples = {
            "run": sampling_config["run"],
            "disk_parameters": disk_parameters,
            "dust_parameters": dust_parameters,
            "planet_parameters": planet_parameters,
            "mesh_parameters": mesh_parameters,
            "extrapolation_parameters": extrapolation_parameters,
            "output_parameters": output_parameters,
            "grid_parameters": grid_parameters,
            "thermal_mc_parameters": thermal_mc_parameters,
            "imaging_parameters": imaging_parameters,
        }

        self._rng = rng
        return samples

    @classmethod
    def new(
        cls,
        num_images: int,
        steps_per_orbit: int,
        num_outputs: int,
        seed: int,
        sim: Simulation,
    ) -> "SimulationRun":
        instance = cls(
            id=sim.get_next_run_id(),
            sim=sim,
            seed=seed,
            num_images=num_images,
            resume_rng=False,
        )
        instance._directory.mkdir(exist_ok=True)
        instance._sampling_config.dump_dict(sim._sampling_config.as_dict())

        instance._sampling_config["run.num_images"] = num_images
        instance._sampling_config["run.seed"] = seed
        instance._sampling_config["run.polar_img_size"] = sim._polar_img_size
        instance._sampling_config["run.steps_per_orbit"] = steps_per_orbit
        instance._sampling_config["run.num_outputs"] = num_outputs
        instance._sampling_config["run.float_type"] = (
            "FLOAT64" if sim._float_type == np.float64 else "FLOAT32"
        )

        return instance


class DiskModel:
    def __init__(self, id: int, run: SimulationRun):
        self._id: int = id
        self._directory: Path = run._directory / f"model_{id}"
        self._run: SimulationRun = run

    def get_dust_density(
        self,
        output_idx: int = -1,
        dust_idx: int = 1,
        extrapolation: bool = False,
        r_scale: str | None = None,
        grid: Grid | None = None,
    ) -> np.ndarray:
        if output_idx < 0:
            output_idx = np.arange(0, self.get_num_outputs())[output_idx]

        density_2d = np.fromfile(
            self.get_data_directory() / f"dust{dust_idx}dens{output_idx}.dat",
            dtype=self._run.get_float_type(),
        ).reshape(self.get_polar_size(extrapolation=False))

        interpolate = (
            (grid is not None and grid.r_scale == "log")
            or r_scale == "log"
            or (
                self.get_sample_config()["grid_parameters.r_scale"] == "log"
                and r_scale is None
            )
        )

        if grid is None:
            grid = self.get_grid(extrapolation=extrapolation, r_scale=r_scale)

        if extrapolation and grid is not None:
            samples = self.get_sample_config().as_dict()
            max_value_radius = (
                np.abs(
                    grid.r_min
                    - (samples["mesh_parameters"]["y_min"] * un.AU).to(un.meter)
                )
                * samples["extrapolation_parameters"]["r_rim_maxium_factor"]
            ).to(un.meter)
            print(max_value_radius.to(un.AU))
            N_r_extrapolated, N_phi = self.get_polar_size(extrapolation=True)
            N_r_non_extrapolated, _ = self.get_polar_size(extrapolation=False)

            N_add_cells = N_r_extrapolated - N_r_non_extrapolated

            density_extrapolated = np.zeros((N_r_extrapolated, N_phi))
            density_extrapolated[N_add_cells:, :] = density_2d

            cutoff_idx = samples["extrapolation_parameters"]["extrapolation_cutoff_idx"]

            x = grid._radii.linear[: N_add_cells + cutoff_idx]

            x_ref = np.concatenate(
                [
                    np.array(
                        [
                            grid.r_min.value,
                            max_value_radius.value,
                        ]
                    ),
                    grid._radii.linear[N_add_cells : N_add_cells + cutoff_idx].value,
                ]
            )

            for col in range(N_phi):
                density_slice = density_2d[:, col]

                max_value_density = (
                    density_slice[0]
                    * samples["extrapolation_parameters"]["density_rim_extend_factor"]
                )
                min_value_density = density_slice.min()

                y_ref = np.concatenate(
                    [
                        np.array(
                            [
                                min_value_density,
                                max_value_density,
                            ]
                        ),
                        density_slice[:cutoff_idx],
                    ]
                )
                density_extrapolated[:N_add_cells, col] = PchipInterpolator(
                    x_ref, y_ref
                )(x)[:N_add_cells]

            density_2d = density_extrapolated

        if interpolate:
            # Logarithmic interpolation created with the help of GPT-5.2-Codex
            r_centers = 0.5 * (grid._r_edges.log[1:] + grid._r_edges.log[:-1])

            density_2d_log = np.empty((r_centers.size, density_2d.shape[1]))
            for j in range(density_2d.shape[1]):
                density_2d_log[:, j] = np.interp(
                    r_centers.value, grid._radii.linear.value, density_2d[:, j]
                )

            return density_2d_log
        else:
            return density_2d

    def get_dust_density_3d(
        self,
        output_idx: int,
        r_scale: str | None,
        grid: Grid,
        extrapolation: bool,
        dust_idx: int = 1,
    ) -> np.ndarray:
        unit_system = self._run._sim._unit_system

        density_2d = (
            self.get_dust_density(
                output_idx=output_idx,
                dust_idx=dust_idx,
                r_scale=r_scale,
                grid=grid,
                extrapolation=extrapolation,
            )
            * unit_system.mass
            / unit_system.length**2
        ).si  # kg / m^2

        # 2d -> 3d relations from https://www.aanda.org/articles/aa/pdf/2009/12/aa11220-08.pdf

        zs = grid.radii[None].T @ np.cos(grid.thetas.value)[None]
        height_ratios = zs**2 / (
            2 * np.tile(grid.heights[:, None], (1, grid.N_theta)) ** 2
        )

        samples = self.get_sample_config()

        stokes_number = 1 / samples[f"dust_parameters.invstokes.{dust_idx}"]

        eps = -stokes_number / diffusion_coefficient(
            stokes_number=stokes_number,
            alpha_viscosity=samples["disk_parameters.alpha"],
        )

        exponent = np.exp(eps * (np.exp(height_ratios) - 1) - height_ratios)

        density_3d = density_2d[:, :, None] * exponent[:, None, :] / un.meter
        density_3d[density_3d.value < 1e-20] = 0
        density_3d = density_3d.swapaxes(1, 2)

        # Normalization code created with the help of Claude Sonnet 4.6
        norm = np.zeros(grid.N_r)
        for i in range(grid.N_r):
            h = grid.heights[i].si.value
            h2 = 2 * h**2

            integrand = lambda z: np.exp(eps * (np.exp(z**2 / h2) - 1) - z**2 / h2)  # noqa U038

            norm[i], _ = integrate.quad(
                integrand, -grid.heights.si.max().value, grid.heights.si.max().value
            )

        density_3d /= norm[:, None, None]

        return density_3d

    def get_polar_dust_density(
        self,
        grid_shape: tuple[int],
        extrapolation: bool,
        output_idx: int = -1,
        dust_idx: int = 1,
        a_maj: float = 1.0,
        b_min: float = 1.0,
        rot_angle: float = 0.0,
        r_scale: str | None = None,
        xy_lims: ArrayLike | None = None,
        xy_unit: un.Unit = un.AU,
    ) -> tuple[np.ndarray, float, float]:
        polar_intensities = self.get_dust_density(
            output_idx=output_idx,
            dust_idx=dust_idx,
            extrapolation=extrapolation,
            r_scale=r_scale,
        )

        grid = self.get_grid(extrapolation=extrapolation, r_scale=r_scale)
        r_min, r_max = grid.r_min.value, grid.r_max.value

        rs, phis = grid.get_polar_grid(r_mode=r_scale)

        return (
            ellipse_img2cartesian_img(
                r=rs.T,
                phi=phis.T,
                intensities=polar_intensities,
                grid_shape=grid_shape,
                a=a_maj,
                b=b_min,
                alpha=rot_angle,
                xy_lims=xy_lims
                if xy_lims is not None
                else [[-r_max, r_max], [-r_max, r_max]],
            ),
            r_min,
            r_max,
        )

    def get_image(
        self, idx: int, fov: float | un.Quantity | None = None
    ) -> un.Quantity | np.ndarray:
        with open(self.get_image_directory() / f"image_{idx}.out") as file:
            img_data = file.readlines()

        for i in range(len(img_data)):
            mod_data = img_data[i].strip()
            if mod_data != "":
                img_data[i] = mod_data

        img_shape = np.array(img_data[1].split(), dtype=int).tolist()
        img = np.array(img_data[6:-1], dtype=np.float64).reshape(img_shape)

        if fov is not None:
            flux_unit = (
                un.erg
                * un.centimeter ** (-2)
                * un.hertz ** (-1)
                * un.second ** (-1)
                * un.steradian ** (-1)
            )

            fov = fov if isinstance(fov, un.Quantity) else fov * un.arcsecond

            solid_angle = 4 * np.arcsin(np.sin(fov / 2) ** 2)
            solid_angle = solid_angle.value * un.steradian

            return (img * flux_unit).to(un.jansky)
        else:
            return img

    def get_dust_temperature(self) -> np.ndarray:
        temperature = np.fromfile(
            self.get_data_directory().parent / "radmc3d/dust_temperature.dat", sep="\n"
        )[3:]
        grid = self.get_grid()
        temperature = temperature.reshape((grid.N_phi, grid.N_theta, grid.N_r)).T
        return temperature * un.Kelvin

    def get_cumulative_mass(
        self, radius: float | ArrayLike | un.Quantity
    ) -> un.Quantity:
        sample_config = self.get_sample_config()
        unit_system = self._run._sim._unit_system

        if not hasattr(radius, "unit"):
            radius = radius * unit_system.length
        else:
            radius = radius.to(unit_system.length)

        s0 = sigma0(
            ref_radius=(sample_config["disk_parameters.disk_mass_ref_radius"] * un.AU)
            .to(unit_system.length)
            .value,
            R0=self._run._sim._constants["R0"].value,
            mass=(sample_config["disk_parameters.disk_mass"] * const.M_sun)
            .to(unit_system.mass)
            .value,
            sigma_slope=sample_config["disk_parameters.sigma_slope"],
        )

        return (
            mass_function(
                radius=radius.value,
                sigma_slope=sample_config["disk_parameters.sigma_slope"],
                sigma0=s0,
                R0=self._run._sim._constants["R0"].value,
            )
            * unit_system.mass
        )

    def get_height(
        self,
        radius: float | ArrayLike | un.Quantity,
        flaring_index: None | float = None,
    ) -> un.Quantity:
        sample_config = self.get_sample_config()
        unit_system = self._run._sim._unit_system

        if not hasattr(radius, "unit"):
            radius = radius * unit_system.length
        else:
            radius = radius.to(unit_system.length)

        if flaring_index is None:
            flaring_index = sample_config["disk_parameters.flaring_index"]

        return (
            disk_height(
                radius=radius.value,
                ref_aspect_ratio=sample_config["disk_parameters.aspect_ratio"],
                flaring_index=flaring_index,
                R0=self._run._sim._constants["R0"].value,
            )
            * unit_system.length
        )

    def get_approximate_grain_size(
        self, dust_idx: int, output_idx: int = -1
    ) -> un.Quantity:
        unit_system = self._run._sim._unit_system

        if output_idx < 0:
            output_idx = np.arange(0, self.get_num_outputs())[output_idx]

        # Average density of grains (solid_dust_density) for DSHARP taken from
        # https://github.com/birnstiel/dsharp_opac/blob/2715ec5a1ebb892cca20737f54f8ccad317c8466/notebooks/opacity_examples.ipynb
        return approximate_grain_size(
            stokes_number=1
            / self.get_sample_config()[f"dust_parameters.invstokes.{dust_idx}"],
            solid_dust_density=1.6686 * un.gram / un.centimeter**3,
            gas_surface_density=np.fromfile(
                self.get_data_directory() / f"gasdens{output_idx}.dat",
                dtype=self._run.get_float_type(),
            ).mean()
            * (1 * unit_system.mass / unit_system.length**2).si,
        ).decompose()

    def get_opacities(
        self,
        dust_idx: int,
        wavelengths: np.ndarray | un.Quantity,
        output_idx: int = -1,
    ) -> dict:
        diel_const, rho_s = opacities.get_dsharp_mix()

        grain_size = self.get_approximate_grain_size(
            dust_idx=dust_idx, output_idx=output_idx
        )

        wavelengths = (
            wavelengths * un.micrometer
            if isinstance(wavelengths, np.ndarray)
            else wavelengths
        )

        return opacities.get_opacities(
            a=np.array([grain_size.to(un.centimeter).value]),
            lam=wavelengths.to(un.centimeter).value,
            diel_const=diel_const,
            rho_s=rho_s,
            extrapol=True,
            extrapolate_large_grains=True,
        )

    def plot_height_profile(
        self,
        flaring_index: None | float = None,
        save_to: str | PathLike | None = None,
        save_args: dict | None = None,
        r_unit: un.Unit = un.AU,
        r_min: float | None = None,
        r_max: float | None = None,
        x_norm: str | None = None,
        y_norm: str | None = None,
        plot_args: dict | None = None,
        fig: matplotlib.figure.Figure | None = None,
        fig_args: dict | None = None,
        ax: matplotlib.axes.Axes | None = None,
    ) -> tuple[matplotlib.axes.Axes, matplotlib.figure.Figure]:
        save_args = {} if save_args is None else save_args
        fig_args = {} if fig_args is None else fig_args
        plot_args = {} if plot_args is None else plot_args

        r_min = r_min if r_min is not None else self.get_radius_lims()[0]
        r_max = r_max if r_max is not None else self.get_radius_lims()[1]

        radii = np.linspace(r_min, r_max, 10000)

        height = self.get_height(radius=radii * un.AU, flaring_index=flaring_index)

        fig, ax = configure_axes(fig=fig, ax=ax)
        sample_config = self.get_sample_config()

        ax.plot(
            (radii * un.AU).to(r_unit).value,
            height.to(r_unit).value,
            label=f"Flaring Index = {sample_config['disk_parameters.flaring_index']}",
        )

        ax.axvline(
            (self.get_radius_lims()[0] * un.AU).to(r_unit).value,
            ls="dashed",
            color="maroon",
            alpha=0.4,
            label="Inner Simulation Radius",
        )
        ax.axvline(
            (self.get_radius_lims()[1] * un.AU).to(r_unit).value,
            ls="dashed",
            color="green",
            alpha=0.4,
            label="Outer Simulation Radius",
        )
        ax.set_xlabel(f"Radius $R$ / {r_unit.to_string(format='latex')}")
        ax.set_ylabel(f"Height H(R) / {r_unit.to_string(format='latex')}")

        if x_norm is not None:
            ax.set_xscale(x_norm)

        if y_norm is not None:
            ax.set_yscale(y_norm)

        ax.legend()

        if save_to is not None:
            fig.savefig(save_to, **save_args)

        return fig, ax

    def plot_cumulative_mass(
        self,
        save_to: str | PathLike | None = None,
        save_args: dict | None = None,
        r_unit: un.Unit = un.AU,
        show_formula: bool = False,
        r_min: float | None = None,
        r_max: float | None = None,
        x_norm: str | None = None,
        y_norm: str | None = None,
        plot_args: dict | None = None,
        fig: matplotlib.figure.Figure | None = None,
        fig_args: dict | None = None,
        ax: matplotlib.axes.Axes | None = None,
    ) -> tuple[matplotlib.axes.Axes, matplotlib.figure.Figure]:
        save_args = {} if save_args is None else save_args
        fig_args = {} if fig_args is None else fig_args
        plot_args = {} if plot_args is None else plot_args

        r_min = r_min if r_min is not None else self.get_radius_lims()[0]
        r_max = r_max if r_max is not None else self.get_radius_lims()[1]

        radii = np.linspace(r_min, r_max, 10000)
        disk_mass = self.get_cumulative_mass(radius=radii * un.AU) / const.M_sun

        fig, ax = configure_axes(fig=fig, ax=ax)

        ax.plot(
            (radii * un.AU).to(r_unit).value,
            disk_mass,
            label=(
                "$M(<R) = \\frac{2\\pi}{2-p}\\Sigma_0 R_0^2 \\cdot"
                "\\left[\\left(\\frac{R}{R_0}\\right)^{2-p}-1\\right]$"
            )
            if show_formula
            else None,
            **plot_args,
        )
        ax.axvline(
            (self.get_radius_lims()[0] * un.AU).to(r_unit).value,
            ls="dashed",
            color="maroon",
            alpha=0.4,
            label="Inner Simulation Radius",
        )
        ax.axvline(
            (self.get_radius_lims()[1] * un.AU).to(r_unit).value,
            ls="dashed",
            color="green",
            alpha=0.4,
            label="Outer Simulation Radius",
        )
        ax.set_xlabel(f"Radius $R$ / {r_unit.to_string(format='latex')}")
        ax.set_ylabel("Cumulative Disk Mass $M(<R)$ / $M_{\\text{sun}}$")

        if x_norm is not None:
            ax.set_xscale(x_norm)

        if y_norm is not None:
            ax.set_yscale(y_norm)

        ax.legend()

        if save_to is not None:
            fig.savefig(save_to, **save_args)

        return fig, ax

    def plot_density_profile(
        self,
        save_to: str | PathLike | None = None,
        save_args: dict | None = None,
        r_unit: un.Unit = un.AU,
        density_unit: un.Unit = un.kilogram / un.meter**2,
        show_formula: bool = False,
        r_min: float | None = None,
        r_max: float | None = None,
        x_norm: str | None = None,
        y_norm: str | None = None,
        plot_args: dict | None = None,
        fig: matplotlib.figure.Figure | None = None,
        fig_args: dict | None = None,
        ax: matplotlib.axes.Axes | None = None,
    ) -> tuple[matplotlib.axes.Axes, matplotlib.figure.Figure]:
        save_args = {} if save_args is None else save_args
        fig_args = {} if fig_args is None else fig_args
        plot_args = {} if plot_args is None else plot_args

        r_min = r_min if r_min is not None else self.get_radius_lims()[0]
        r_max = r_max if r_max is not None else self.get_radius_lims()[1]

        radii = np.linspace(r_min, r_max, 10000)

        sample_config = self.get_sample_config()
        unit_system = self._run._sim._unit_system

        s0 = sigma0(
            ref_radius=(sample_config["disk_parameters.disk_mass_ref_radius"] * un.AU)
            .to(unit_system.length)
            .value,
            R0=self._run._sim._constants["R0"].value,
            mass=(sample_config["disk_parameters.disk_mass"] * const.M_sun)
            .to(unit_system.mass)
            .value,
            sigma_slope=sample_config["disk_parameters.sigma_slope"],
        )

        density = surface_density(
            (radii * un.AU).to(unit_system.length).value,
            R0=self._run._sim._constants["R0"].value,
            sigma0=s0,
            sigma_slope=sample_config["disk_parameters.sigma_slope"],
        ) * (unit_system.mass / unit_system.length**2).to(density_unit)

        fig, ax = configure_axes(fig=fig, ax=ax)

        ax.plot(
            (radii * un.AU).to(r_unit).value,
            density,
            label="$\\Sigma(R)=\\Sigma_0 \\cdot (\\frac{R}{R_0})^{-p}$"
            if show_formula
            else None,
            **plot_args,
        )
        ax.axvline(
            (self.get_radius_lims()[0] * un.AU).to(r_unit).value,
            ls="dashed",
            color="maroon",
            alpha=0.4,
            label="Inner Simulation Radius",
        )
        ax.axvline(
            (self.get_radius_lims()[1] * un.AU).to(r_unit).value,
            ls="dashed",
            color="green",
            alpha=0.4,
            label="Outer Simulation Radius",
        )
        ax.set_xlabel(f"Radius $R$ / {r_unit.to_string(format='latex')}")
        ax.set_ylabel(
            f"Density Profile $\\Sigma$ / {density_unit.to_string(format='latex')}"
        )

        if x_norm is not None:
            ax.set_xscale(x_norm)

        if y_norm is not None:
            ax.set_yscale(y_norm)

        ax.legend()

        if save_to is not None:
            fig.savefig(save_to, **save_args)

        return fig, ax

    def plot_dust_density(
        self,
        grid_shape: tuple,
        extrapolation: bool = False,
        r_scale: str | None = None,
        output_idx: int = -1,
        dust_idx: int = 1,
        a_maj: float = 1.0,
        b_min: float = 1.0,
        rot_angle: float = 0.0,
        xy_lims: ArrayLike | None = None,
        xy_unit: un.Unit = un.AU,
        intensity_limits: ArrayLike | None = None,
        save_to: str | None = None,
        save_args: dict = None,
        **kwargs,
    ) -> tuple[
        matplotlib.image.AxesImage, matplotlib.figure.Figure, matplotlib.axes.Axes
    ]:
        unit_system = self._run._sim._unit_system

        polar_intensities, _, r_max = self.get_polar_dust_density(
            grid_shape=grid_shape,
            extrapolation=extrapolation,
            output_idx=output_idx,
            dust_idx=dust_idx,
            a_maj=a_maj,
            b_min=b_min,
            rot_angle=rot_angle,
            xy_lims=xy_lims,
            xy_unit=xy_unit,
            r_scale=r_scale,
        )

        xy_lims = xy_lims if xy_lims is not None else [[-r_max, r_max], [-r_max, r_max]]
        dens_unit = unit_system.mass / unit_system.length**2

        return plot_image(
            data=polar_intensities,
            xy_lims=xy_lims,
            intensity_label=f"Dust density / {dens_unit.to_string(format='latex')}",
            intensity_limits=intensity_limits,
            dtype=self._run.get_float_type(),
            save_to=save_to,
            save_args=save_args,
            **kwargs,
        )

    def animate_dust_density(
        self,
        grid_shape: tuple,
        step_size: int,
        extrapolation: bool = False,
        r_scale: str | None = None,
        output_fmt: str = "mp4",
        output_dir: str | PathLike | None = None,
        save_to: str | PathLike | None = None,
        save_with_timestamp: bool = False,
        dust_idx: int = 1,
        start_idx: int = 0,
        a_maj: float = 1.0,
        b_min: float = 1.0,
        rot_angle: float = 0.0,
        end_idx: int | None = None,
        xy_unit: un.Unit = un.AU,
        save_args: dict = None,
        fps: int = 30,
        dpi: int | str = "figure",
        blit: bool = True,
        show_progress: bool = True,
        **kwargs,
    ) -> None:
        if save_to is not None:
            save_to = Path(save_to)
        else:
            output_dir = Path(output_dir)
            save_to = (
                output_dir / f"{self._run._sim.name}-run_{self._run._id}-"
                f"model_{self._id}.{output_fmt}"
            )

        print(
            "Animation length will be: "
            f"{np.round(self.get_num_outputs() // step_size / fps, 2)} seconds"
        )

        end_idx = self.get_num_outputs() if end_idx is None else end_idx
        num_outputs = (end_idx - start_idx) // step_size

        data = np.zeros((num_outputs, *grid_shape))

        output_idcs = np.arange(start_idx, end_idx + step_size, step=step_size)

        for i in np.arange(0, num_outputs):
            img, _, _ = self.get_polar_dust_density(
                grid_shape=grid_shape,
                extrapolation=extrapolation,
                r_scale=r_scale,
                output_idx=output_idcs[i],
                dust_idx=dust_idx,
                a_maj=a_maj,
                b_min=b_min,
                rot_angle=rot_angle,
                xy_unit=xy_unit,
            )
            data[i] = img

        im, fig, ax = self.plot_dust_density(
            grid_shape=grid_shape,
            extrapolation=extrapolation,
            r_scale=r_scale,
            output_idx=start_idx,
            dust_idx=dust_idx,
            xy_unit=xy_unit,
            a_maj=a_maj,
            b_min=b_min,
            rot_angle=rot_angle,
            intensity_limits=[data[data > 0].min(), data.max()],
            **kwargs,
        )

        def update(frame: int):
            im.set_data(data[frame + 1])
            return [im]

        anim = animation.FuncAnimation(
            fig=fig, func=update, frames=num_outputs - 1, blit=blit, interval=1e3 / fps
        )

        writer = None
        if save_to.suffix.lower() == ".gif":
            writer = animation.PillowWriter(
                fps=fps,
                bitrate=-1,
            )
            writer.setup(fig=fig, outfile=save_to, dpi=dpi)

        def _progress_func(_i, _n):
            progress_bar.update(1)

        with tqdm(
            total=num_outputs - 1, desc="Saving animation", disable=not show_progress
        ) as progress_bar:
            if save_with_timestamp:
                save_to = save_to.with_stem(
                    f"{save_to.stem}-{Time(time(), format='unix').isot}."
                )
            if writer is None:
                anim.save(save_to, progress_callback=_progress_func, dpi=dpi)
            else:
                anim.save(
                    save_to, progress_callback=_progress_func, writer=writer, dpi=dpi
                )

    def exists(self) -> bool:
        return self._directory.exists()

    def get_num_species(self) -> int:
        return len(self.get_sample_config()["dust_parameters.invstokes"])

    def get_polar_size(self, extrapolation: bool) -> tuple[int]:
        polar_size = self._run.get_polar_img_size()
        if not extrapolation:
            return polar_size
        else:
            samples = self.get_sample_config()

            polar_size = list(polar_size)
            r_min, r_max = self.get_radius_lims()
            r_cell_size = np.abs(r_max - r_min) / polar_size[0]

            r_min_extrapolated = samples["extrapolation_parameters"]["r_min"]

            polar_size[0] += int(np.abs(r_min - r_min_extrapolated) // r_cell_size)
            return tuple(polar_size)

    def get_grid(
        self,
        extrapolation: bool,
        r_scale: str | None = None,
        theta_scale: str | None = None,
    ) -> Grid:
        samples = self.get_sample_config()

        return Grid(
            model=self,
            r_scale=samples["grid_parameters"]["r_scale"]
            if r_scale is None
            else r_scale,
            extrapolation=extrapolation,
            theta_steps=samples["grid_parameters"]["theta_steps"],
            theta_scale=samples["grid_parameters"]["theta_scale"]
            if theta_scale is None
            else theta_scale,
            theta_log_exp=samples["grid_parameters"]["theta_log_exp"],
            theta_tol=samples["grid_parameters"]["theta_tol"],
        )

    def get_time_deltas(self) -> float:
        sample_config = self.get_sample_config()
        planet_parameters = sample_config["planet_parameters"]

        distances = np.array(planet_parameters["planet_orbit_radius"]) * un.AU

        num_orbits = sample_config["output_parameters.num_largest_orbits"]

        period = orbital_period(
            mass=np.sum(planet_parameters["stellar_mass"]) * const.M_sun,
            radius=distances.max().to(self._run._sim._unit_system.length),
            G=self._run._sim._constants["G"],
        )

        total_time = num_orbits * period

        return (
            period / self._run.get_steps_per_orbit(),
            total_time / self.get_num_outputs(),
        )

    def get_num_outputs(self) -> int:
        files = {
            f
            for f in Path(self.get_data_directory()).glob("gasdens*.dat")
            if f.is_file() and "2d" not in f.name
        }
        return len(files)

    def get_num_images(self) -> int:
        return int(self.get_sample_config()["imaging_parameters.num_versions"])

    def get_radius_lims(self, extrapolation: bool = False) -> tuple[float]:
        sample_config = self.get_sample_config()

        r_min = (
            sample_config["mesh_parameters.y_min"]
            if not extrapolation
            else sample_config["extrapolation_parameters.r_min"]
        )
        r_max = (
            np.max(sample_config["planet_parameters.planet_orbit_radius"])
            * sample_config["mesh_parameters.y_max_ratio"]
        )

        return r_min, r_max

    def get_sample_config(self) -> TOMLConfiguration:
        return TOMLConfiguration(self._directory / "samples.toml")

    def is_extrapolation_active(self) -> bool:
        return self.get_sample_config()["extrapolation_parameters.extrapolation_active"]

    def get_image_directory(self) -> Path:
        return self._directory.resolve() / "images"

    def get_data_directory(self) -> Path:
        return self._directory.resolve() / "data"

    def get_radmc3d_directory(self) -> Path:
        return self._directory.resolve() / "radmc3d"

    def get_fargo_output_path(self) -> str:
        return f"outputs/sim_{self._run._sim.name}/run_{self._run._id}/model_{self._id}"

    def delete(self) -> None:
        shutil.rmtree(self._directory)

    def get_rng(self) -> np.random.Generator:
        rng_dump = self._directory / "rng_state.pkl"

        with open(rng_dump, "rb") as pkl:
            return pickle.load(pkl)

    def get_execution_times(self) -> dict:
        record_toml = TOMLConfiguration(
            self.get_data_directory() / "execution_time.toml"
        )

        if not record_toml.is_valid():
            raise FileNotFoundError(
                "The execution times were not recored for this model."
            )

        return record_toml.as_dict()

    @classmethod
    def new(cls, id: int, run: SimulationRun):
        instance = cls(id=id, run=run)
        instance._directory.mkdir(exist_ok=True)
        return instance
