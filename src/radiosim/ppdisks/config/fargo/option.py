import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from radiosim.ppdisks.config import Parser, Variables

__all__ = [
    "FargoOptionConfig",
    "FargoOptionEntry",
]


class OptionType(Enum):
    VARIABLE = " = "
    LIST = " := "
    OPTION = " += "

    def __init__(self, seperator: str):
        self.seperator: str = seperator

    def split_line(self, line: str) -> tuple[str]:
        if line.endswith("\n"):
            line = line.replace("\n", "")

        key, value = line.split(self.seperator)

    @classmethod
    def from_value(cls, value: str) -> "OptionType":
        for option_type in cls.__members__.values():
            if option_type.seperator in value:
                return option_type

        raise TypeError("The given value is no valid OptionType!")


@dataclass
class FargoOptionEntry:
    value: object | None
    option_type: OptionType
    enabled: bool = True

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def get_line(self, key: str) -> str:
        match self.option_type:
            case OptionType.VARIABLE:
                return f"{key} = {self.value}"
            case OptionType.LIST:
                if not isinstance(self.value, list):
                    raise ValueError(
                        "The OptionType LIST has to contain a value with type list!"
                    )
                return f"{key} := {' '.join(self.value)}"
            case OptionType.OPTION:
                if self.value is None:
                    return f"FARGO_OPT += -D{key}"
                else:
                    return f"FARGO_OPT += -D{key}={self.value}"

    @classmethod
    def option(enabled: bool, value: object | None = None) -> "FargoOptionEntry":
        return FargoOptionEntry(
            value=value,
            option_type=OptionType.OPTION,
            enabled=enabled,
        )


class FargoOptionConfig:
    def __init__(self, setup: str, autosave: bool = False):
        self._path: Path = Variables.get("FARGO_ROOT") / f"setups/{setup}/{setup}.opt"
        self._autosave: bool = autosave

        if not self._path.exists():
            raise NameError(f"The given setup '{setup}' does not exist!")

        # initialize default parameters
        self._parameters: dict = {
            "fluids": {
                # This parameter automatically implies the list variable FLUIDS
                # and the option NFLUIDS
                "NFLUIDS": FargoOptionEntry(value="2", option_type=OptionType.VALUE),
                "DRAGFORCE": FargoOptionEntry.option(enabled=True),
                "STOKESNUMBER": FargoOptionEntry.option(enabled=True),
                "DUSTDIFFUSION": FargoOptionEntry.option(enabled=True),
                "COLLISIONPREDICTOR": FargoOptionEntry.option(enabled=False),
            },
            "performance": {"FLOAT": FargoOptionEntry.option(enabled=True)},
            "dimensions": {
                "X": FargoOptionEntry.option(enabled=True),
                "Y": FargoOptionEntry.option(enabled=True),
                "Z": FargoOptionEntry.option(enabled=False),
            },
            "coordinates": {
                "CARTESIAN": FargoOptionEntry.option(enabled=False),
                "CYLINDRICAL": FargoOptionEntry.option(enabled=True),
                "SPHERICAL": FargoOptionEntry.option(enabled=False),
            },
            "planetary_system": {
                "NODEFAULTSTAR": FargoOptionEntry.option(enabled=False),
            },
            "equation_of_state": {
                "ADIABATIC": FargoOptionEntry.option(enabled=False),
                "ISOTHERMAL": FargoOptionEntry.option(enabled=True),
            },
            "additional_physics": {
                "MHD": FargoOptionEntry.option(enabled=False),
                "STRICTSYM": FargoOptionEntry.option(enabled=False),
                "OHMICDIFFUSION": FargoOptionEntry.option(enabled=False),
                "AMBIPOLARDIFFUSION": FargoOptionEntry.option(enabled=False),
                "HALLEFFECT": FargoOptionEntry.option(enabled=False),
                "VISCOSITY": FargoOptionEntry.option(enabled=False),
                "ALPHAVISCOSITY": FargoOptionEntry.option(enabled=True),
                "POTENTIAL": FargoOptionEntry.option(enabled=True),
                "STOCKHOLM": FargoOptionEntry.option(enabled=True),
                "HILLCUT": FargoOptionEntry.option(enabled=False),
            },
            "shearing_box": {
                "SHEARINGBOX": FargoOptionEntry.option(enabled=False),
                "SHEARINGBC": FargoOptionEntry.option(enabled=False),
            },
            "transport": {
                "RAM": FargoOptionEntry.option(enabled=False),
                "STANDARD": FargoOptionEntry.option(enabled=False),
            },
            "slopes": {
                "DONOR": FargoOptionEntry.option(enabled=False),
            },
            "artificial_viscosity": {
                "NOSUBSTEP2": FargoOptionEntry.option(enabled=False),
                "STRONG_SHOCK": FargoOptionEntry.option(enabled=False),
            },
            "boundaries": {
                "HARDBOUNDARIES": FargoOptionEntry.option(enabled=False),
            },
            "utils": {
                "LONGSUMMARY": FargoOptionEntry.option(enabled=True),
            },
            "cuda_blocks": {
                "BLOCK_X": FargoOptionEntry.option(value=16, enabled=True),
                "BLOCK_Y": FargoOptionEntry.option(value=16, enabled=True),
                "BLOCK_Z": FargoOptionEntry.option(value=1, enabled=True),
            },
        }

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
                    if len(category_dict > 0):
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
                                FargoOptionEntry(
                                    value=list(range(value.value)),
                                    option_type=OptionType.LIST,
                                    enabled=True,
                                ).get_line(key="FLUIDS")
                            )
                            lines.append(value.get_line(key=key))
                            lines.append(
                                FargoOptionEntry.option(
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
                entry = self[key]
                if entry.option_type != option_type:
                    raise TypeError(
                        f"The variable '{key}' has to have the type "
                        f"{entry.option_type}! (In config: {option_type})"
                    )

                entry.value = Parser().parse(value)
                entry.enable()

    def __getitem__(self, key: str) -> FargoOptionEntry | dict:
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
                if isinstance(value, FargoOptionEntry):
                    self._parameters[key_components[0]][key_components[1]] = value
                elif isinstance(
                    self._parameters[key_components[0]][key_components[1]],
                    FargoOptionEntry,
                ):
                    self._parameters[key_components[0]][key_components[1]].value = value
                else:
                    raise TypeError(
                        "This key does not point to a valid entry! Enter an instance "
                        "of a 'FargoOptionEntry'"
                    )
            case _:
                if len(key_components) > 2:
                    raise KeyError(
                        "The maximum depth of a config key is 2 (catgeory -> entry)!"
                    )

        if self._autosave:
            self.save()
