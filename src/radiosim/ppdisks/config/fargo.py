import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from radiosim.ppdisks.config import Parser, Variables

__all__ = [
    "FargoParameterConfig",
    "FargoParameterEntry",
    "FargoOptionConfig",
    "FargoOptionEntry",
]


@dataclass
class FargoParameterEntry:
    key: str
    value: object
    comment: str

    def get_line(self, max_key_len: int, max_value_len: int):
        return (
            f"{self.key:<{max_key_len + 2}}{self.value:<{max_value_len + 2}}"
            f"{self.comment if self.comment is not None else ''}\n"
        )


class FargoParameterConfig:
    def __init__(self, setup: str, autosave: bool = False):
        self._path: Path = Variables.get("FARGO_ROOT") / f"setups/{setup}/{setup}.par"
        self._autosave: bool = autosave

        if not self._path.exists():
            raise NameError(f"The given setup '{setup}' does not exist!")

        self._parameters: dict[str, FargoParameterEntry] = dict()

        self.load()
        if self._parameters["Setup"].value != setup:
            warnings.warn(
                "The given setup name exists but the 'Setup' parameter in the"
                " config gives a different name. A missmatch might lead to "
                "execution problems.",
                stacklevel=1,
            )

    def _get_entries(self) -> list[FargoParameterEntry]:
        values = []
        for _key, value in self._parameters.items():
            if isinstance(value, dict):
                values.extend(list(value.values()))
            else:
                values.append(value)

        return values

    def load(self) -> None:
        with open(self._path) as file:
            lines = file.readlines()

            current_category = None
            for line in lines:
                if line.strip() == "":
                    continue

                if line.startswith("### ") and "[" in line and "]" in line:
                    current_category = (
                        line.removeprefix("### ").split("[")[1].split("]")[0]
                    )
                    self._parameters[current_category] = dict()
                    continue

                components = line.split()

                if len(components) < 2:
                    continue

                entry = FargoParameterEntry(
                    key=components[0],
                    value=Parser().parse(components[1]),
                    comment=None if len(components) == 2 else " ".join(components[2:]),
                )

                if current_category is None:
                    self._parameters[components[0]] = entry
                else:
                    self._parameters[current_category][components[0]] = entry

    def save(self) -> None:
        with open(self._path) as file:
            old_content = file.read()
        with open(self._path, "w") as file:
            try:
                key_lens = []
                value_lens = []
                for entry in self._get_entries():
                    key_lens.append(len(str(entry.key)))
                    value_lens.append(len(str(entry.value)))

                max_key_len = np.max(key_lens)
                max_value_len = np.max(value_lens)

                lines = []

                for key, entry in self._parameters.items():
                    if isinstance(entry, dict):
                        lines.append("\n")
                        lines.append(f"### [{key}]\n")
                        lines.append("\n")

                        for _subkey, subentry in self._parameters[key].items():
                            lines.append(
                                subentry.get_line(
                                    max_key_len=max_key_len, max_value_len=max_value_len
                                )
                            )
                    else:
                        lines.append(
                            entry.get_line(
                                max_key_len=max_key_len, max_value_len=max_value_len
                            )
                        )

                file.writelines(lines)
            except Exception as e:
                warnings.warn(
                    "An error occured while saving. Rolling back configuration files.",
                    stacklevel=1,
                )
                file.write(old_content)
                raise e

    def __getitem__(self, key: str) -> FargoParameterEntry:
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
                if isinstance(value, FargoParameterEntry):
                    self._parameters[key_components[0]] = value
                elif isinstance(
                    self._parameters[key_components[0]], FargoParameterEntry
                ):
                    self._parameters[key_components[0]].value = value
                else:
                    raise TypeError(
                        "Values at root level must either be a dict or a valid entry!"
                    )
            case 2:
                if isinstance(value, FargoParameterEntry):
                    self._parameters[key_components[0]][key_components[1]] = value
                elif isinstance(
                    self._parameters[key_components[0]][key_components[1]],
                    FargoParameterEntry,
                ):
                    self._parameters[key_components[0]][key_components[1]].value = value
                else:
                    raise TypeError(
                        "This key does not point to a valid entry! Enter an instance "
                        "of a 'FargoParameterEntry'"
                    )

            case _:
                if len(key_components) > 2:
                    raise KeyError(
                        "The maximum depth of a config key is 2 (catgeory -> entry)!"
                    )

        if self._autosave:
            self.save()


class OptionType(Enum):
    VARIABLE = 1
    LIST = 2
    OPTION = 3


@dataclass
class FargoOptionEntry:
    value: object | None
    option_type: OptionType
    enabled: bool = True
    cuda_only: bool = False

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    @classmethod
    def option(
        enabled: bool, value: object | None = None, cuda_only: bool = False
    ) -> "FargoOptionEntry":
        return FargoOptionEntry(
            value=value,
            option_type=OptionType.OPTION,
            enabled=enabled,
            cuda_only=cuda_only,
        )


class FargoOptionConfig:
    def __init__(self):
        self._parameters = {
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
            "cuda_blocks": {
                "BLOCK_X": FargoOptionEntry.option(
                    value=16, enabled=True, cuda_only=True
                ),
                "BLOCK_Y": FargoOptionEntry.option(
                    value=16, enabled=True, cuda_only=True
                ),
                "BLOCK_Z": FargoOptionEntry.option(
                    value=1, enabled=True, cuda_only=True
                ),
            },
        }
