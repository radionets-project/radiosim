import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from ..parser import Parser
from ..variables import Variables

__all__ = [
    "FargoOptionConfig",
    "OptionEntry",
]


class OptionType(Enum):
    VARIABLE = " = "
    LIST = " := "
    OPTION = " += "

    def __init__(self, seperator: str):
        self.seperator: str = seperator

    def __repr__(self) -> str:
        return str(self)

    def split_line(self, line: str) -> tuple[str, None | str]:
        if line.endswith("\n"):
            line = line.replace("\n", "")

        if self == OptionType.OPTION:
            line = line.split(self.seperator)[1]
            line = line.replace("-D", "")
            components = line.split("=")
            key = components[0]
            value = None if len(components) == 1 else components[1]
        else:
            key, value = line.split(self.seperator)

        return key, value

    @classmethod
    def from_value(cls, value: str) -> "OptionType":
        for option_type in cls.__members__.values():
            if option_type.seperator in value:
                return option_type

        raise TypeError(f"The given value '{value}' is no valid OptionType!")


@dataclass
class OptionEntry:
    value: object | None
    option_type: OptionType
    enabled: bool = True

    def __repr__(self) -> str:
        return f"(value={self.value}, type={self.option_type}, enabled={self.enabled})"

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def get_line(self, key: str) -> str:
        match self.option_type:
            case OptionType.VARIABLE:
                return f"{key} = {self.value}\n"
            case OptionType.LIST:
                if not isinstance(self.value, list):
                    raise ValueError(
                        "The OptionType LIST has to contain a value with type list!"
                    )
                value = [str(i) for i in self.value]
                return f"{key} := {' '.join(value)}\n"
            case OptionType.OPTION:
                if self.value is None:
                    return f"FARGO_OPT += -D{key}\n"
                else:
                    return f"FARGO_OPT += -D{key}={self.value}\n"

    @classmethod
    def option(cls, enabled: bool, value: object | None = None) -> "OptionEntry":
        return OptionEntry(
            value=value,
            option_type=OptionType.OPTION,
            enabled=enabled,
        )


class FargoOptionConfig:
    def __init__(self, setup: str, autosave: bool = False):
        self._path: Path = Variables.get("FARGO_ROOT") / f"setups/{setup}/{setup}.opt"
        self._autosave: bool = autosave

        if not self._path.parent.exists():
            raise NameError(f"The given setup '{setup}' does not exist!")

        # initialize default parameters
        self._parameters: dict = {
            "fluids": {
                # This parameter automatically implies the list variable FLUIDS
                # and the option NFLUIDS
                "NFLUIDS": OptionEntry(value=2, option_type=OptionType.VARIABLE),
                "DRAGFORCE": OptionEntry.option(enabled=True),
                "STOKESNUMBER": OptionEntry.option(enabled=True),
                "DUSTDIFFUSION": OptionEntry.option(enabled=True),
                "COLLISIONPREDICTOR": OptionEntry.option(enabled=False),
            },
            "performance": {"FLOAT": OptionEntry.option(enabled=True)},
            "dimensions": {
                "X": OptionEntry.option(enabled=True),
                "Y": OptionEntry.option(enabled=True),
                "Z": OptionEntry.option(enabled=False),
            },
            "coordinates": {
                "CARTESIAN": OptionEntry.option(enabled=False),
                "CYLINDRICAL": OptionEntry.option(enabled=True),
                "SPHERICAL": OptionEntry.option(enabled=False),
            },
            "planetary_system": {
                "NODEFAULTSTAR": OptionEntry.option(enabled=False),
            },
            "equation_of_state": {
                "ADIABATIC": OptionEntry.option(enabled=False),
                "ISOTHERMAL": OptionEntry.option(enabled=True),
            },
            "additional_physics": {
                "MHD": OptionEntry.option(enabled=False),
                "STRICTSYM": OptionEntry.option(enabled=False),
                "OHMICDIFFUSION": OptionEntry.option(enabled=False),
                "AMBIPOLARDIFFUSION": OptionEntry.option(enabled=False),
                "HALLEFFECT": OptionEntry.option(enabled=False),
                "VISCOSITY": OptionEntry.option(enabled=False),
                "ALPHAVISCOSITY": OptionEntry.option(enabled=True),
                "POTENTIAL": OptionEntry.option(enabled=True),
                "STOCKHOLM": OptionEntry.option(enabled=True),
                "HILLCUT": OptionEntry.option(enabled=False),
            },
            "shearing_box": {
                "SHEARINGBOX": OptionEntry.option(enabled=False),
                "SHEARINGBC": OptionEntry.option(enabled=False),
            },
            "transport": {
                "RAM": OptionEntry.option(enabled=False),
                "STANDARD": OptionEntry.option(enabled=False),
            },
            "slopes": {
                "DONOR": OptionEntry.option(enabled=False),
            },
            "artificial_viscosity": {
                "NOSUBSTEP2": OptionEntry.option(enabled=False),
                "STRONG_SHOCK": OptionEntry.option(enabled=False),
            },
            "boundaries": {
                "HARDBOUNDARIES": OptionEntry.option(enabled=False),
            },
            "utils": {
                "LONGSUMMARY": OptionEntry.option(enabled=True),
            },
            "cuda_blocks": {
                "BLOCK_X": OptionEntry.option(value=16, enabled=True),
                "BLOCK_Y": OptionEntry.option(value=16, enabled=True),
                "BLOCK_Z": OptionEntry.option(value=1, enabled=True),
            },
        }

        if not self._path.exists():
            self.save()
        else:
            self.load()

    def disable_all(self):
        for _category, category_dict in self._parameters.items():
            for _key, value in category_dict.items():
                value.disable()

    def save(self):
        with open(self._path) as file:
            old_content = file.read()

        with open(self._path, "w") as file:
            try:
                lines = []
                for category, category_dict in self._parameters.items():
                    if np.sum([entry.enabled for entry in category_dict.values()]) == 0:
                        continue

                    lines.append("\n")
                    lines.append(f"# [{category}]\n")
                    lines.append("\n")

                    if category == "cuda_blocks":
                        lines.append("ifeq (${GPU}, 1)\n")

                    for key, value in category_dict.items():
                        if not value.enabled:
                            continue

                        if key == "NFLUIDS":
                            lines.append(
                                OptionEntry(
                                    value=list(range(value.value)),
                                    option_type=OptionType.LIST,
                                    enabled=True,
                                ).get_line(key="FLUIDS")
                            )
                            lines.append(value.get_line(key=key))
                            lines.append(
                                OptionEntry.option(
                                    value="${NFLUIDS}", enabled=True
                                ).get_line(key="NFLUIDS")
                            )
                        else:
                            lines.append(value.get_line(key=key))

                    if category == "cuda_blocks":
                        lines.append("endif")

                file.writelines(lines)

            except Exception as e:
                warnings.warn(
                    "An error occured while saving. Rolling back configuration files.",
                    stacklevel=1,
                )
                file.write(old_content)
                raise e

    def load(self):
        self.disable_all()
        with open(self._path) as file:
            lines = file.readlines()

            current_category = None
            for line in lines:
                if line.strip() == "":
                    continue

                if line.startswith("ifeq (${GPU}, 1)") or line.startswith("endif"):
                    continue

                if line.startswith("# ") and "[" in line and "]" in line:
                    current_category = (
                        line.removeprefix("# ").split("[")[1].split("]")[0]
                    )
                    continue

                if current_category is None:
                    raise ValueError(
                        "The sections of the .opt file are not valid! "
                        "Every variable must be inside a catgeory!"
                    )

                option_type = OptionType.from_value(line)
                key, value = option_type.split_line(line=line)

                if key == "FLUIDS" or (
                    key == "NFLUIDS" and option_type == OptionType.OPTION
                ):
                    continue

                entry = self[f"{current_category}.{key}"]

                if entry.option_type != option_type:
                    raise TypeError(
                        f"The variable '{key}' has to have the type "
                        f"{entry.option_type}! (In config: {option_type})"
                    )

                entry.value = Parser().parse(value)
                entry.enable()

    def info(self, enabled: bool = False):
        def dict2str(d: dict, indent: int = 0, increase: int = 4):
            out = ""
            for key, value in d.items():
                if isinstance(value, dict):
                    out += f"{key}:\n"
                    out += dict2str(value, indent=indent + increase, increase=increase)
                else:
                    if value.enabled or not enabled:
                        out += " " * indent + f"'{key}': {value}\n"

            return out

        return dict2str(d=self._parameters)

    def __repr__(self):
        return self.info(enabled=False)

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, key: str) -> OptionEntry | dict:
        key_components = key.split(".")

        match len(key_components):
            case 1:
                return self._parameters[key_components[0]]
            case 2:
                return self._parameters[key_components[0]][key_components[1]]
            case _:
                if len(key_components) > 2:
                    raise KeyError(
                        "The maximum depth of a config key is 2 (catgeory -> entry)!"
                    )

    def __setitem__(self, key: str, value: object) -> None:
        key_components = key.split(".")

        match len(key_components):
            case 1:
                if isinstance(value, dict):
                    self._parameters[key_components[0]] = value
                    return None
                else:
                    raise TypeError("Values at root level must either be a dict!")
            case 2:
                if isinstance(value, OptionEntry):
                    self._parameters[key_components[0]][key_components[1]] = value
                elif isinstance(
                    self._parameters[key_components[0]][key_components[1]],
                    OptionEntry,
                ):
                    self._parameters[key_components[0]][key_components[1]].value = value
                else:
                    raise TypeError(
                        "This key does not point to a valid entry! Enter an instance "
                        "of a 'OptionEntry'"
                    )
            case _:
                if len(key_components) > 2:
                    raise KeyError(
                        "The maximum depth of a config key is 2 (categeory -> entry)!"
                    )

        if self._autosave:
            self.save()
