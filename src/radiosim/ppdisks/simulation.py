import pickle
import shutil
from os import PathLike
from pathlib import Path

import numpy as np

from radiosim.ppdisks.config import TOMLConfiguration

from .config.fargo import Constants, PlanetConfig, UnitSystem
from .setup import Setup

__all__ = ["Simulation"]

_distributions = {"uniform"}


def get_default_sampling_config():
    return {
        "disk_parameters": {
            "aspect_ratio": [0.01, 0.1],
            "sigma0": [1.0, 200.0],
            "sigma_slope": [0.05, 0.3],
            "flaring_index": [0.0, 0.0],
            "alpha": [0.001, 0.01],  # Shakura-Sunyaev viscosity parameter
        },
        "dust_parameters": {
            "invstokes": {
                "1": [8.0, 20.0],
            },
            "epsilon": [0.05, 0.2],
        },
        "planet_parameters": {
            "binary_ratio": 0.667,  # Ratio of binary systems to single systems
            "binary_period": [6.04800e5, 3e9],  # Seconds (logarithmic sampling)
            "stellar_mass": [0.5, 5],  # Solar Masses
            "number_of_planets": [1, 5],
            "planet_mass": [1.0e-6, 5.0e-3],  # Solar Masses
            "planet_orbit_radius": [7.0, 30.0],  # Astronomical Units
            "eccentricity": [0.0, 0.2],  # 0 = Circle, 0 < e < 1 = Ellipse
        },
        "mesh_parameters": {
            "y_min": [4.0, 6.0],  # Astronomical Units
            "y_max_ratio": [1.2, 3],  # Multiple of max(orbital_radius)
        },
        "output_parameters": {
            "num_largest_orbits": [600, 800],
        },
    }


class Simulation:
    def __init__(
        self,
        name: str,
        root_directory: PathLike,
        setup: Setup,
        float_type: type,
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

        self._unit_system: UnitSystem = unit_system
        self._constants: Constants = Constants(unit_system=unit_system)

        self._planet_config: PlanetConfig = PlanetConfig(
            name=f"radiosim_{self.name}", autosave=False, unit_system=self._unit_system
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
                "unit_system": self._unit_system.name,
            }
        }
        self._config.dump_dict(content)

    def get_next_run_id(self) -> int:
        dirs = [d for d in self._out_directory.glob("run_*") if d.is_dir()]
        if len(dirs) == 0:
            return 0

        dir_ids = [int(str(d).removeprefix("run_")) for d in dirs]
        return np.max(dir_ids) + 1

    def simulate(
        self,
        seed: int | None = None,
        run_id: int | None = None,
        model_id: int | None = None,
    ) -> None:
        pass

    @classmethod
    def new(
        cls,
        name: str,
        setup: str,
        sampling_config: PathLike | dict | None,
        parent_directory: PathLike | None = None,
        float_type: type = np.float64,
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
            unit_system=unit_system,
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
            unit_system=UnitSystem.__members__[config["general.unit_system"]],
        )

        return instance


class SimulationRun:
    def __init__(self, id: int, sim: Simulation):
        self._id: int = id
        self._directory: Path = sim._out_directory / f"run_{self._id}"

        self._sampling_config: TOMLConfiguration = TOMLConfiguration(
            self._directory / "sampling_config.toml",
            create_if_not_exists=True,
        )

        self._models: list[DiskModel] = [
            DiskModel(id=int(str(d).removeprefix("model_")), run=self)
            for d in self._directory.glob("model_*")
            if d.is_dir()
        ]

    def get_seed(self) -> int:
        return self._sampling_config["seed"]

    def get_model(self, id: int) -> "DiskModel":
        for model in self._models:
            if model._id == id:
                return model

        raise KeyError(f"There is no disk model with id '{id}' in run '{self._id}'.")

    def get_next_model_id(self) -> int:
        dirs = [d for d in self._directory.glob("model_*") if d.is_dir()]
        if len(dirs) == 0:
            return 0

        dir_ids = [int(str(d).removeprefix("model_")) for d in dirs]
        return np.max(dir_ids) + 1

    def get_rng(self, model_id: int | None = None) -> np.random.Generator:
        if model_id is None:
            return np.random.default_rng(seed=self.get_seed())

        return self.get_model(id=model_id).get_rng()

    def draw_samples(self) -> dict:
        sampling_config = self._sampling_config.as_dict()

        def sample_dict(read_dict):
            write_dict = dict()
            for key, value in read_dict.items():
                if key == "seed":
                    continue

                if isinstance(value, dict):
                    write_dict[key] = sample_dict(read_dict=value)
                elif isinstance(value, list):
                    if isinstance(value[0], int):
                        write_dict[key] = self._rng.integers(
                            low=value[0], high=value[1]
                        )
                    else:
                        write_dict[key] = self._rng.uniform(low=value[0], high=value[1])
                else:
                    match key:
                        case "binary_ratio":
                            write_dict["binary_system"] = (
                                self._rng.uniform(0, 1) <= value
                            )

            return write_dict

        return sample_dict(read_dict=sampling_config)

    def new(cls, seed: int, sim: Simulation) -> "SimulationRun":
        instance = cls(id=sim.get_next_run_id(), sim=sim)
        instance._directory.mkdir(exists_ok=True)
        instance._sampling_config.dump_dict(sim._sampling_config.as_dict())
        instance._sampling_config["seed"] = seed

        return instance


class DiskModel:
    def __init__(self, id: int, run: SimulationRun):
        self._id: int = id
        self._directory: Path = run._directory / f"model_{id}"

    def get_rng(self) -> np.random.Generator:
        rng_dump = self._directory / "rng_state.pkl"

        with open(rng_dump, "rb") as pkl:
            return pickle.load(pkl)
