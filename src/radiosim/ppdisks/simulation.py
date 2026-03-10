import pickle
import shutil
from os import PathLike
from pathlib import Path

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy import units as un
from tqdm.auto import tqdm

from radiosim.ppdisks.config import TOMLConfiguration
from radiosim.ppdisks.plotting.plotting import plot_polar_image

from .config import Variables
from .config.fargo import Constants, Planet, PlanetConfig, UnitSystem
from .setup import Setup

__all__ = ["Simulation", "SimulationRun", "DiskModel"]


def get_default_sampling_config():
    return {
        "disk_parameters": {
            "aspect_ratio": [0.01, 0.1],  # Disk aspect ratio @ r=R0 (default R0 = 1AU)
            "sigma0": [15000.0, 30000.0],  # Surface density (kg/m^2) @ r=R0
            "sigma_slope": [0.05, 0.3],  # Exponent of the density profile
            "flaring_index": [0.0, 0.0],
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
            "stellar_temperature": [2000.0, 3000.0],  # Kelvin
            "num_planets": [1, 4],
            "planet_mass": [1.0e-6, 5.0e-3],  # Solar Masses
            "planet_orbit_radius": [6.0, 15.0],  # Astronomical Units
            # short: PEF -> no other planets allowed closer than R_orbit * PEF
            "planet_exclusion_factor": 0.2,
            "eccentricity": [0.0, 0.9],  # 0 = Circle, 0 < e < 1 = Ellipse
        },
        "mesh_parameters": {
            "y_min": [3.0, 5.0],  # Astronomical Units
            "y_max_ratio": [1.5, 3],  # Multiple of max(orbital_radius)
        },
        "output_parameters": {
            "num_largest_orbits": [100, 200],
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
        unit_system: UnitSystem,
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

        self._polar_img_size: tuple[int] = polar_img_size

        self._unit_system: UnitSystem = unit_system
        self._constants: Constants = Constants(unit_system=unit_system, autosave=True)

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
                "polar_img_size": list(self._polar_img_size),
                "unit_system": self._unit_system.name,
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
        num_models: int,
        seed: int,
        num_outputs: int | None = None,
        steps_per_orbit: int | None = None,
        run_id: int | None = None,
        resume: bool = True,
        gpu: bool = True,
        cuda_device_id: int = 0,
        parallel: bool = False,
        num_nodes: int = 1,
        show_progress: bool = True,
        record_execution_time: bool = True,
        verbose: bool = False,
        overwrite: bool = False,
    ) -> None:
        if run_id is None:
            run = SimulationRun.new(
                num_models=num_models,
                steps_per_orbit=steps_per_orbit,
                num_outputs=num_outputs,
                seed=seed,
                sim=self,
            )
        else:
            run = SimulationRun(id=run_id, sim=self, resume_rng=resume)

        print(f"------ STARTING RUN {run._id} ------")

        start_idx = 0 if not resume else run.get_next_model_id()
        for i in np.arange(start_idx, num_models):
            try:
                model = run.get_model(id=i)

                if overwrite:
                    print(
                        f"WARNING! The model with id '{i}' already exists. "
                        "It will be overwritten"
                    )
                    model.delete()
                else:
                    print(
                        f"WARNING! The model with id '{i}' already exists. "
                        "It will not be overwritten. If you want to overwrite it, "
                        "set overwrite=True!"
                    )
                    continue
            except KeyError:
                pass

            model = DiskModel.new(id=i, run=run)
            samples = run.draw_samples()
            run.save_rng(model_id=model._id)

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

            param_config["disk_parameters.aspectRatio"] = disk_parameters[
                "aspect_ratio"
            ]
            param_config["disk_parameters.sigma0"] = disk_parameters["sigma0"]
            param_config["disk_parameters.sigmaSlope"] = disk_parameters["sigma_slope"]
            param_config["disk_parameters.flaringIndex"] = disk_parameters[
                "flaring_index"
            ]
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
            param_config["output_parameters.outputDir"] = str(
                model.get_fargo_output_path()
            )

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

            # Dump samples to TOML file

            sample_dump = samples.copy()

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
            sample_config.dump_dict(content=toml_serialize_dict(read_dict=sample_dump))

            # Recompile and Run Setup

            compile_time = self._setup.compile(
                gpu=gpu,
                parallel=parallel,
                unit_system=self._unit_system,
                rescale=False,
                model_id=model._id,
                show_progress=show_progress,
                verbose=verbose,
                show_fargo_output=verbose,
                return_execution_time=record_execution_time,
            )

            run_time = self._setup.run(
                model_id=model._id,
                num_nodes=num_nodes,
                parallel=parallel,
                show_progress=show_progress,
                cuda_device_id=cuda_device_id,
                verbose=verbose,
                return_execution_time=record_execution_time,
            )

            # Move the data files to the correct directory
            model.get_data_directory().mkdir()
            for file in (
                Variables.get("FARGO_ROOT") / model.get_fargo_output_path()
            ).glob("*.*"):
                shutil.move(
                    src=file,
                    dst=model.get_data_directory(),
                )
            shutil.rmtree(
                path=Variables.get("FARGO_ROOT") / model.get_fargo_output_path()
            )

            if record_execution_time:
                record_toml = TOMLConfiguration(
                    path=model.get_data_directory() / "execution_time.toml",
                    create_if_not_exists=True,
                )
                record_toml.dump_dict(
                    {
                        "compile_time": compile_time,
                        "run_time": run_time[0],
                        "output_times": run_time[1],
                    }
                )

    @classmethod
    def new(
        cls,
        name: str,
        setup: str,
        sampling_config: PathLike | dict | None,
        parent_directory: PathLike | None = None,
        float_type: type = np.float64,
        polar_img_size: tuple[int] = (300, 800),
        unit_system: UnitSystem | str = UnitSystem.MKS,
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
            polar_img_size=polar_img_size,
            unit_system=unit_system.default(),
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
            polar_img_size=tuple(config["general.polar_img_size"]),
            unit_system=UnitSystem.__members__[config["general.unit_system"]],
        )

        return instance


class SimulationRun:
    def __init__(
        self,
        id: int,
        sim: Simulation,
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

        if seed is None:
            self._rng: np.random.Generator = np.random.default_rng(seed=self.get_seed())
        else:
            self._rng: np.random.Generator = np.random.default_rng(seed=seed)

        if resume_rng:
            model_id = self.get_next_model_id() - 1

            if model_id < 0:
                model_id = None

            self._rng = self.get_rng(model_id=model_id)

    def get_polar_img_size(self) -> tuple[int]:
        return tuple(self._sampling_config["polar_img_size"])

    def get_num_outputs(self) -> int:
        return self._sampling_config["num_outputs"]

    def get_steps_per_orbit(self) -> int:
        return self._sampling_config["steps_per_orbit"]

    def get_float_type(self) -> type:
        return (
            np.float64
            if self._sampling_config["float_type"] == "FLOAT64"
            else np.float32
        )

    def get_seed(self) -> int:
        return self._sampling_config["seed"]

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
        model = self.get_model(id=self.get_next_model_id() - 1 if None else model_id)
        rng_path = model._directory / "rng_state.pkl"

        with open(rng_path, "wb") as pkl:
            pickle.dump(self._rng, pkl, pickle.HIGHEST_PROTOCOL)

    def draw_samples(self, model_id: int | None = None) -> dict:
        sampling_config = self._sampling_config.as_dict()
        rng = self.get_rng(model_id=model_id)

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

            while not valid:
                orbit = rng.uniform(
                    low=planet_sampling["planet_orbit_radius"][0],
                    high=planet_sampling["planet_orbit_radius"][1],
                )

                # if new orbit violates planet exclusion zones around other planets
                if np.any(
                    np.abs(planet_orbits[:i_planet] - orbit) / planet_orbits[:i_planet]
                    > planet_sampling["planet_exclusion_factor"]
                ):
                    continue
                else:
                    valid = True

            planet_orbits[i_planet] = orbit

        planet_parameters["planet_orbit_radius"] = planet_orbits

        planet_parameters["eccentricity"] = rng.uniform(
            low=planet_sampling["eccentricity"][0],
            high=planet_sampling["eccentricity"][1],
        )

        mesh_parameters = sample_dict(sampling_config["mesh_parameters"])
        output_parameters = sample_dict(sampling_config["output_parameters"])

        samples = {
            "disk_parameters": disk_parameters,
            "dust_parameters": dust_parameters,
            "planet_parameters": planet_parameters,
            "mesh_parameters": mesh_parameters,
            "output_parameters": output_parameters,
        }

        return samples

    @classmethod
    def new(
        cls,
        num_models: int,
        steps_per_orbit: int,
        num_outputs: int,
        seed: int,
        sim: Simulation,
    ) -> "SimulationRun":
        instance = cls(
            id=sim.get_next_run_id(),
            sim=sim,
            seed=seed,
            resume_rng=False,
        )
        instance._directory.mkdir(exist_ok=True)
        instance._sampling_config.dump_dict(sim._sampling_config.as_dict())

        instance._sampling_config["seed"] = seed
        instance._sampling_config["polar_img_size"] = sim._polar_img_size
        instance._sampling_config["steps_per_orbit"] = steps_per_orbit
        instance._sampling_config["num_outputs"] = num_outputs
        instance._sampling_config["float_type"] = (
            "FLOAT64" if sim._float_type == np.float64 else "FLOAT32"
        )

        return instance


class DiskModel:
    def __init__(self, id: int, run: SimulationRun):
        self._id: int = id
        self._directory: Path = run._directory / f"model_{id}"
        self._run: SimulationRun = run

    def get_num_outputs(self) -> int:
        files = {
            f
            for f in Path(self.get_data_directory()).glob("gasdens*.dat")
            if f.is_file() and "2d" not in f.name
        }
        return len(files)

    def get_dust_density(self, output_idx: int = -1, dust_idx: int = 1) -> np.ndarray:
        if output_idx < 0:
            output_idx = np.arange(0, self.get_num_outputs())[output_idx]

        return np.fromfile(
            self.get_data_directory() / f"dust{dust_idx}dens{output_idx}.dat",
            dtype=self._run.get_float_type(),
        ).reshape(self._run.get_polar_img_size())

    def plot_dust_density(
        self,
        grid_shape: tuple,
        output_idx: int = -1,
        dust_idx: int = 1,
        xy_unit: un.Unit = un.AU,
        save_to: str | None = None,
        save_args: dict = None,
        **kwargs,
    ) -> tuple[
        matplotlib.image.AxesImage, matplotlib.figure.Figure, matplotlib.axes.Axes
    ]:
        r_min, r_max = self.get_radius_lims()
        r_min = (r_min * un.AU).to(xy_unit).value
        r_max = (r_max * un.AU).to(xy_unit).value

        unit_system = self._run._sim._unit_system

        return plot_polar_image(
            polar_intensities=self.get_dust_density(
                output_idx=output_idx, dust_idx=dust_idx
            ),
            grid_shape=grid_shape,
            r_lims=[r_min, r_max],
            intensity_unit=(
                "Dust density / "
                f"{
                    (unit_system.mass / unit_system.length**2).to_string(format='latex')
                }"
            ),
            dtype=self._run.get_float_type(),
            save_to=save_to,
            save_args=save_args,
            **kwargs,
        )

    def animate_dust_density(
        self,
        grid_shape: tuple,
        save_to: str | PathLike,
        step_size: int,
        dust_idx: int = 1,
        xy_unit: un.Unit = un.AU,
        save_args: dict = None,
        fps: int = 30,
        dpi: int | str = "figure",
        blit: bool = True,
        show_progress: bool = True,
        **kwargs,
    ) -> None:
        save_to = Path(save_to)

        ims = []

        fig, ax = plt.subplots()

        print(
            "Animation length will be: "
            f"{np.round(self.get_num_outputs() // step_size / fps, 2)} seconds"
        )

        for i in tqdm(
            np.arange(start=0, stop=self.get_num_outputs(), step=step_size),
            desc="Plotting densities",
            disable=not show_progress,
        ):
            im, _, _ = self.plot_dust_density(
                grid_shape=grid_shape,
                output_idx=i,
                dust_idx=dust_idx,
                xy_unit=xy_unit,
                fig=fig,
                ax=ax,
                **kwargs,
            )
            ims.append([im])
            if hasattr(im, "colorbar") and im.colorbar is not None:
                im.colorbar.ax.remove()

        anim = animation.ArtistAnimation(fig, ims, blit=blit, interval=1e3 / fps)

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
            total=len(ims), desc="Saving animation", disable=not show_progress
        ) as progress_bar:
            if writer is None:
                anim.save(save_to, progress_callback=_progress_func, dpi=dpi)
            else:
                anim.save(
                    save_to, progress_callback=_progress_func, writer=writer, dpi=dpi
                )

    def get_radius_lims(self) -> tuple[float]:
        sample_config = self.get_sample_config()

        r_min = sample_config["mesh_parameters.y_min"]
        r_max = (
            np.max(sample_config["planet_parameters.planet_orbit_radius"])
            * sample_config["mesh_parameters.y_max_ratio"]
        )

        return r_min, r_max

    def get_sample_config(self) -> TOMLConfiguration:
        return TOMLConfiguration(self._directory / "samples.toml")

    def get_data_directory(self) -> Path:
        return self._directory.resolve() / "data"

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
