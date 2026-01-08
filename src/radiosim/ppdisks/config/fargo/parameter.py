import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..parser import Parser
from ..variables import Variables

__all__ = [
    "FargoParameterConfig",
    "ParameterEntry",
]


@dataclass
class ParameterEntry:
    key: str
    value: object
    comment: str

    def __repr__(self) -> str:
        comment_val = f"'{self.comment}'" if self.comment is not None else self.comment
        return f"(value={self.value}, comment={comment_val})"

    def get_line(self, max_key_len: int, max_value_len: int):
        match self.value:
            case float():
                value = str(self.value)
                if "e" in value:
                    num, exp = value.split("e")
                    value = f"{float(num)}e{int(exp)}"

            case bool():
                value = "Yes" if self.value else "No"
            case _:
                value = self.value

        return (
            f"{self.key:<{max_key_len + 2}}{value:<{max_value_len + 2}}"
            f"{self.comment if self.comment is not None else ''}"
        ).strip() + "\n"


class FargoParameterConfig:
    def __init__(self, setup: str, autosave: bool = False):
        self._path: Path = Variables.get("FARGO_ROOT") / f"setups/{setup}/{setup}.par"
        self._autosave: bool = autosave

        if not self._path.exists():
            raise NameError(f"The given setup '{setup}' does not exist!")

        self._parameters: dict[str, ParameterEntry] = dict()

        self.load()
        if self._parameters["Setup"].value != setup:
            warnings.warn(
                "The given setup name exists but the 'Setup' parameter in the"
                " config gives a different name. A missmatch might lead to "
                "execution problems.",
                stacklevel=1,
            )

    def _get_entries(self) -> list[ParameterEntry]:
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

                entry = ParameterEntry(
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

    def __repr__(self):
        def dict2str(d: dict, indent: int = 0, increase: int = 4):
            out = ""
            for key, value in d.items():
                if isinstance(value, dict):
                    out += f"{key}:\n"
                    out += dict2str(value, indent=indent + increase, increase=increase)
                else:
                    out += " " * indent + f"'{key}': {value}\n"

            return out

        return dict2str(d=self._parameters)

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, key: str) -> ParameterEntry:
        key_components = key.split(".")

        match len(key_components):
            case 1:
                return self._parameters[key_components[0]]
            case 2:
                return self._parameters[key_components[0]][key_components[1]]
            case _:
                if len(key_components) > 2:
                    raise KeyError(
                        "The maximum depth of a config key is 2 (categeory -> entry)!"
                    )

    def __setitem__(self, key: str, value: object) -> None:
        key_components = key.split(".")

        match len(key_components):
            case 1:
                if isinstance(value, dict):
                    self._parameters[key_components[0]] = value
                    return None
                if isinstance(value, ParameterEntry):
                    self._parameters[key_components[0]] = value
                elif isinstance(self._parameters[key_components[0]], ParameterEntry):
                    self._parameters[key_components[0]].value = value
                else:
                    raise TypeError(
                        "Values at root level must either be a dict or a valid entry!"
                    )
            case 2:
                if isinstance(value, ParameterEntry):
                    self._parameters[key_components[0]][key_components[1]] = value
                elif isinstance(
                    self._parameters[key_components[0]][key_components[1]],
                    ParameterEntry,
                ):
                    self._parameters[key_components[0]][key_components[1]].value = value
                else:
                    raise TypeError(
                        "This key does not point to a valid entry! Enter an instance "
                        "of a 'ParameterEntry'"
                    )

            case _:
                if len(key_components) > 2:
                    raise KeyError(
                        "The maximum depth of a config key is 2 (categeory -> entry)!"
                    )

        if self._autosave:
            self.save()
