import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import sympy
from astropy import constants
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
    key: str
    mass: un.Unit
    length: un.Unit
    time: un.Unit
    temperature: un.Unit
    electric_current: un.Unit


class UnitSystem(UnitSystemData, Enum):
    SCALE_FREE = (
        "SF",
        "0",
        un.dimensionless_unscaled,
        un.dimensionless_unscaled,
        un.dimensionless_unscaled,
        un.dimensionless_unscaled,
        un.dimensionless_unscaled,
    )
    MKS = "MKS", "MKS", un.kilogram, un.meter, un.second, un.Kelvin, un.ampere
    CGS = "CGS", "CGS", un.gram, un.centimeter, un.second, un.Kelvin, un.cgs.abA

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
    def __init__(
        self, unit_system: UnitSystem = UnitSystem.MKS, autosave: bool = False
    ):
        self._path: Path = Variables.get("FARGO_ROOT") / "src/fondam.h"
        self._unit_system: UnitSystem = unit_system
        self._constants: ConstantSet = ConstantSet()
        self._autosave: bool = autosave

        if not self._path.exists():
            raise FileNotFoundError(
                f"The default constant file could not be "
                f"found at {self._path}. This file is required to run FARGO3D. "
                "Check your FARGO3D installation!"
            )

        self.load()

    def save(self):
        with open(self._path) as file:
            lines = file.readlines()

        original_content = lines.copy()

        max_key_length = 6 + len(self._unit_system.suffix)

        with open(self._path, "w") as file:
            try:
                for i in range(len(lines)):
                    line = lines[i]

                    if not line.startswith("#define"):
                        continue

                    line = line.replace("#define", "").strip()

                    components = line.split("//")[0].split()

                    if not components[0].endswith(self._unit_system.suffix):
                        continue

                    key, _ = components
                    value = getattr(
                        self._constants,
                        key.removesuffix(f"_{self._unit_system.suffix}"),
                    ).value

                    value = str(value)
                    if "e" in value:
                        num, exp = value.split("e")
                        value = f"{float(num)}e{int(exp)}"

                    lines[i] = f"#define  {key:>{max_key_length}}  {value}\n"

                file.writelines(lines)
            except Exception as e:
                warnings.warn(
                    "An error occured while saving. Rolling back configuration files.",
                    stacklevel=1,
                )
                file.writelines(original_content)
                raise e

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

    def __repr__(self):
        return "\n".join(
            [f"{key} = {const}" for key, const in self._constants.__dict__.items()]
        )

    def __str__(self):
        return self.__repr__()

    def __setitem__(self, key: str, value: float | un.Quantity):
        constant = self[key]
        if isinstance(value, float):
            setattr(self._constants, key, value * constant.unit)
        elif isinstance(value, un.Quantity):
            setattr(self._constants, key, value.to(constant.unit))
        else:
            raise TypeError(
                "The constant must either be a float value "
                "or an astropy.units.Quantity!"
            )

        if self._autosave:
            self.save()

    def __getitem__(self, key: str) -> un.Quantity:
        return getattr(self._constants, key)

    def reset(self) -> None:
        self = self.default(unit_system=self._unit_system)

    @classmethod
    def default(cls, unit_system: UnitSystem):
        instance = cls(unit_system=unit_system)
        match instance._unit_system:
            case UnitSystem.SCALE_FREE:
                instance._constants = ConstantSet(
                    G=1.0 * instance._unit_system.get_unit(constant="G"),
                    MSTAR=1.0 * instance._unit_system.get_unit(constant="MSTAR"),
                    R0=1.0 * instance._unit_system.get_unit(constant="R0"),
                    R_MU=1.0 * instance._unit_system.get_unit(constant="R_MU"),
                    MU0=1.0 * instance._unit_system.get_unit(constant="MU0"),
                )
            case UnitSystem.MKS:
                instance._constants = ConstantSet(
                    G=constants.G.value * instance._unit_system.get_unit(constant="G"),
                    MSTAR=constants.M_sun.value
                    * instance._unit_system.get_unit(constant="MSTAR"),
                    R0=constants.au.value
                    * instance._unit_system.get_unit(constant="R0"),
                    R_MU=3460.0 * instance._unit_system.get_unit(constant="R_MU"),
                    MU0=constants.mu0.value
                    * instance._unit_system.get_unit(constant="MU0"),
                )
            case UnitSystem.CGS:
                instance._constants = ConstantSet(
                    G=constants.G.cgs.value
                    * instance._unit_system.get_unit(constant="G"),
                    MSTAR=constants.M_sun.cgs.value
                    * instance._unit_system.get_unit(constant="MSTAR"),
                    R0=constants.au.cgs.value
                    * instance._unit_system.get_unit(constant="R0"),
                    R_MU=36149835.0 * instance._unit_system.get_unit(constant="R_MU"),
                    MU0=12.5663706143591
                    * instance._unit_system.get_unit(constant="MU0"),
                )

        return instance
