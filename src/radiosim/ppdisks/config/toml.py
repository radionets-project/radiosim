import tomllib
from pathlib import Path

import tomli_w


def _parse_toml_key(key: str) -> list[str]:
    key_components = key.split(".")
    return key_components


class InvalidTOMLConfigurationError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class TOMLConfiguration:
    def __init__(
        self,
        path: str | Path,
        create_if_not_exists: bool = False,
        none_if_unknown_key: bool = False,
    ):
        self._path: Path = Path(path) if isinstance(path, str) else path
        self._none_if_unknown_key: bool = none_if_unknown_key

        if self._path.suffix != ".toml":
            raise InvalidTOMLConfigurationError(
                "The given configuration file has to be a TOML file!"
            )

        if create_if_not_exists:
            self.create()

    def is_valid(self) -> bool:
        return self._path.is_file() and self._path.suffix == ".toml"

    def __getitem__(self, item: str):
        if not self.is_valid():
            raise InvalidTOMLConfigurationError(
                "The given configuration file is not valid!"
            )

        keys = _parse_toml_key(item)

        with open(self._path, "rb") as tomlf:
            content_dict = tomllib.load(tomlf)

        content = content_dict
        for key in keys:
            if isinstance(content, dict):
                try:
                    content = content[key]
                except KeyError as error:
                    if self._none_if_unknown_key:
                        return None
                    else:
                        raise error
            else:
                raise KeyError(
                    f"The key component '{key}' is set to a non-dict value and "
                    "therefore there cannot be a child value!"
                )

        return content

    def __setitem__(self, key: str, value: object):
        if not self.is_valid():
            raise InvalidTOMLConfigurationError(
                "The variable configuration could not "
                f"be found at location {str(self._path)}!"
            )

        keys = _parse_toml_key(key)
        with open(self._path, "rb") as tomlf:
            content_dict = tomllib.load(tomlf)

        content = content_dict
        for i in range(len(keys)):
            key = keys[i]
            if i < len(keys) - 1:
                if key not in content:
                    content[key] = dict()
                    content = content[key]
                else:
                    if isinstance(content[key], dict):
                        content = content[key]
                    else:
                        raise KeyError(
                            f"The key component '{key}' is already "
                            "set to a non-dict value!"
                        )
            else:
                content[key] = value

        with open(self._path, "wb") as file:
            tomli_w.dump(content_dict, file)

    def __contains__(self, item: str):
        try:
            value = self[item]

            if value is None:
                return False

        except KeyError:
            return False
        return True

    def create(self, create_parents: bool = True) -> None:
        if create_parents:
            self._path.parent.mkdir(exist_ok=True, parents=True)

        self._path.touch(exist_ok=True)

    def dump_dict(self, content: dict) -> None:
        with open(self._path, "wb") as file:
            tomli_w.dump(content, file)

    def as_dict(self) -> dict:
        with open(self._path, "rb") as tomlf:
            return tomllib.load(tomlf)

    def get_keys(self, non_dict_only: bool = False):
        def recursive_keys(dictionary: dict, parent: str | None = None) -> list[str]:
            keys = []
            for key, value in dictionary.items():
                parent_key = f"{parent}.{key}" if parent is not None else key
                if isinstance(value, dict):
                    if not non_dict_only:
                        keys.append(key)
                    keys.extend(recursive_keys(dictionary=value, parent=parent_key))
                else:
                    keys.append(parent_key)
            return keys

        return recursive_keys(dictionary=self.as_dict())

    def get_path(self) -> Path:
        return self._path
