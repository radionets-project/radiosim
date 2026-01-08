from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import sympy
from astropy import units as un

from ..variables import Variables


@dataclass
class ConstantSet:
    G: float = 1.0
    MSTAR: float = 1.0
    R0: float = 1.0
    R_MU: float = 1.0
    MU0: float = 1.0


@dataclass
class UnitSystemData:
    suffix: str
    mass: un.Unit
    length: un.Unit
    time: un.Unit
    temperature: un.Unit
    electric_current: un.Unit


class UnitSystem(UnitSystemData, Enum):
    SCALE_FREE = (
        "SF",
        un.dimensionless_unscaled,
        un.dimensionless_unscaled,
        un.dimensionless_unscaled,
        un.dimensionless_unscaled,
        un.dimensionless_unscaled,
    )
    MKS = "MKS", un.kilogram, un.meter, un.second, un.Kelvin, un.ampere
    CGS = "CGS", un.gram, un.centimeter, un.second, un.Kelvin, 10 * un.ampere

    def get_unit(self, constant: str):
        match constant:
            case "G":
                return self.mass ** (-1) * self.length**3 * self.time ** (-2)
            case "MSTAR":
                return self.mass
            case "R0":
                return self.length
            case "R_MU":
                return self.length**2 * self.time ** (-2) * self.temperature ** (-1)
            case "MU0":
                return (
                    self.mass
                    * self.length
                    * self.time ** (-2)
                    * self.electric_current ** (-2)
                )


class Constants:
    def __init__(self, unit_system: UnitSystem = UnitSystem.SCALE_FREE):
        self._path: Path = Variables.get("FARGO_ROOT") / "src/fondam.h"
        self._unit_system: UnitSystem = unit_system
        self._constants: ConstantSet = ConstantSet()

        if not self._path.exists():
            raise FileNotFoundError(
                f"The default constant file could not be "
                f"found at {self._path}. This file is required to run FARGO3D. "
                "Check your FARGO3D installation!"
            )

        self.load()

    def load(self):
        with open(self._path) as file:
            lines = file.readlines()

            for line in lines:
                if not line.startswith("#define"):
                    continue

                line = line.replace("#define", "").strip()

                components = line.split("//")[0].split()

                if not components[0].endswith(self._unit_system.suffix):
                    continue

                key, value = components

                value = float(sympy.sympify(value.removeprefix("(").removesuffix(")")))

                attribute = key.removesuffix(f"_{self._unit_system.suffix}")
                setattr(
                    self._constants,
                    attribute,
                    value * self._unit_system.get_unit(constant=attribute),
                )
