from pathlib import Path

from .toml import TOMLConfiguration

_DEFAULT_CONFIG_CONTENTS = {}


class MetaConfig:
    def __init__(self):
        self._config: TOMLConfiguration = TOMLConfiguration(
            path=Path.home() / ".radiosim/config.toml",
            create_if_not_exists=None,
            none_if_unknown_key=False,
        )

        if not self._config._path.exists():
            self._config.create(create_parents=True)
            self._config.dump_dict(content=_DEFAULT_CONFIG_CONTENTS)

    def get_config(self) -> TOMLConfiguration:
        return self._config
