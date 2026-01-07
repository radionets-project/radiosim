import shutil
from dataclasses import dataclass
from pathlib import Path

from radiosim.ppdisks.config import Variables


@dataclass
class Planet:
    name: str
    distance: float
    mass: float
    feels_disk: bool
    feels_others: bool
    accretion: float = 0.0

    def get_config_line(self) -> str:
        return (
            f"{self.name} {self.distance} {self.mass} {self.accretion} "
            f"{'YES' if self.feels_disk else 'NO'} "
            f"{'YES' if self.feels_others else 'NO'}"
        )


class PlanetConfig:
    def __init__(self, name: str):
        self.name: str = name
        self.planets: dict = dict()
        self._path: Path = Variables.get("FARGO_ROOT") / f"planets/{name}.cfg"

        if self._path.is_file():
            self.load()

    def add_planet(self, planet: Planet):
        self.planets[planet.name] = planet

    def load(self):
        if not self._path.is_file():
            raise FileNotFoundError(
                f"The Planet configuration '{self.name}' could not be found!"
            )

        with open(self._path) as file:
            lines = file.readlines()

            planets = []
            for line in lines:
                if line.startswith("#") or line == "\n":
                    continue

                try:
                    vals = line.split()
                    planets.append(
                        Planet(
                            name=vals[0],
                            distance=float(vals[1]),
                            mass=float(vals[2]),
                            accretion=float(vals[3]),
                            feels_disk=vals[4] == "YES",
                            feels_others=vals[5] == "YES",
                        )
                    )
                except Exception:
                    continue

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

        shutil.copy(
            self._path,
        )
        return PlanetConfig(name=new_name)

    def _get_content(self) -> list[str]:
        lines = [
            "##############################################",
            "#   Planetary System Initial Configuration   #",
            "##############################################",
            "",
            "# Planet Name\tDistance\tMass\tAccretion\tFeels Disk\tFeels Others",
        ]
        for planet in self.planets:
            lines.append(planet.get_config_line())

    def __repr__(self):
        return f"PlanetConfig(name={self.name}, planets={list(self.planets.keys())})"

    @classmethod
    def get_configs(cls):
        return [
            PlanetConfig(name=file.stem)
            for file in (Variables.get("FARGO_ROOT") / "planets").glob("*.cfg")
            if file.is_file()
        ]
