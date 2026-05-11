from pathlib import Path

from radiosim.ppdisks.config.meta import MetaConfig


class Variables:
    def __init__(self):
        self._variables = {
            "FARGO_ROOT": Path(MetaConfig().get_config()["FARGO_ROOT"]),
            "RADMC3D_ROOT": Path(MetaConfig().get_config()["RADMC3D_ROOT"]),
        }

    @classmethod
    def get_variables(cls) -> dict:
        instance = cls()
        return instance._variables

    @classmethod
    def get(cls, key: str) -> object:
        instance = cls()
        return instance._variables[key]
