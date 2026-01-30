import shutil
import warnings
from pathlib import Path

from astropy import units as un

from ..variables import Variables
from .constants import Constants, UnitSystem

__all__ = ["Planet", "PlanetConfig"]


class Planet:
    def __init__(
        self,
        name: str,
        distance: float | un.Quantity,
        mass: float | un.Quantity,
        feels_disk: bool,
        feels_others: bool,
        unit_system: UnitSystem = UnitSystem.MKS,
        rescale: bool = False,
        accretion: float = 0.0,
    ):
        constants = Constants(unit_system=unit_system)

        self.name: str = name
        self.distance: float = (
            distance
            if isinstance(distance, float)
            else (distance / (constants["R0"] if rescale else 1)).decompose().value
        )
        self.mass: float = (
            mass
            if isinstance(mass, float)
            else (mass / (constants["MSTAR"] if rescale else 1)).decompose().value
        )
        self.feels_disk: bool = feels_disk
        self.feels_others: bool = feels_others
        self.accretion: float = accretion

        self._unit_system: UnitSystem = unit_system
        self._rescaled: bool = rescale

    def __repr__(self):
        return (
            f"Planet(name={self.name}, distance={self.distance}, mass={self.mass}, "
            f"feels_disk={self.feels_disk}, feels_others={self.feels_others}, "
            f"accretion={self.accretion})"
        )

    def __str__(self):
        return self.__repr__()

    def get_dict(self) -> dict:
        return {
            "distance": self.distance,
            "mass": self.mass,
            "feels_disk": self.feels_disk,
            "feels_others": self.feels_others,
            "accretion": self.accretion,
            "unit_system": self._unit_system.name,
            "rescaled": self._rescaled,
        }

    def get_config_line(self) -> str:
        return (
            f"{self.name} {self.distance} {self.mass} {self.accretion} "
            f"{'YES' if self.feels_disk else 'NO'} "
            f"{'YES' if self.feels_others else 'NO'}\n"
        )


class PlanetConfig:
    def __init__(
        self,
        name: str,
        autosave: bool = False,
        unit_system: UnitSystem = UnitSystem.MKS,
    ):
        self.name: str = name
        self.planets: dict = dict()
        self._path: Path = Variables.get("FARGO_ROOT") / f"planets/{name}.cfg"
        self._autosave: bool = autosave
        self._unit_system: UnitSystem = unit_system

        if self._path.is_file():
            self.load()

    def get_toml_dict(self) -> dict:
        dump = {}
        for key, value in self.planets.items():
            dump[key] = value.get_dict()
        return dump

    def add_planet(self, planet: Planet):
        self.planets[planet.name] = planet

        if self._autosave:
            self.save()

    def remove_planet(self, name: str):
        del self.planets[name]

        if self._autosave:
            self.save()

    def load(self):
        if not self._path.is_file():
            raise FileNotFoundError(
                f"The Planet configuration '{self.name}' could not be found!"
            )

        with open(self._path) as file:
            lines = file.readlines()

            planets = []
            for line in lines:
                if line.startswith("#") or line.strip() == "":
                    continue

                vals = line.split()
                if not vals[0][0].isalpha():
                    warnings.warn(
                        "A planet's name must begin with analphanumeric character!",
                        stacklevel=1,
                    )

                planets.append(
                    Planet(
                        name=vals[0],
                        distance=float(vals[1]),
                        mass=float(vals[2]),
                        accretion=float(vals[3]),
                        feels_disk=vals[4] == "YES",
                        feels_others=vals[5] == "YES",
                        unit_system=self._unit_system,
                    )
                )

            for planet in planets:
                self.add_planet(planet)

    def save(self):
        with open(self._path, "w") as file:
            file.writelines(self._get_content())

    def copy(self, new_name: str) -> "PlanetConfig":
        if new_name == self.name:
            raise NameError("The new name may not be equal to the current name!")

        new_path = self._path.parent / f"{new_name}.cfg"

        if new_path.exists():
            raise FileExistsError("This configuration already exists!")

        shutil.copy(self._path, new_path)
        return PlanetConfig(name=new_name)

    def _get_content(self) -> list[str]:
        lines = [
            "##############################################\n",
            "#   Planetary System Initial Configuration   #\n",
            "##############################################\n",
            "\n",
            "# Planet Name\tDistance\tMass\tAccretion\tFeels Disk\tFeels Others\n",
        ]
        for planet in self.planets.values():
            lines.append(planet.get_config_line())
        return lines

    def __repr__(self):
        return f"PlanetConfig(name={self.name}, planets={list(self.planets.keys())})"

    @classmethod
    def get_configs(cls):
        return [
            PlanetConfig(name=file.stem)
            for file in (Variables.get("FARGO_ROOT") / "planets").glob("*.cfg")
            if file.is_file()
        ]
