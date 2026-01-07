from pathlib import Path

from radiosim.ppdisk.config.fargopy import FargopyConfiguration


class Variables:
    def __init__(self):
        self._variables = {
            "FARGO_ROOT": Path(FargopyConfiguration()["FP_FARGO3D_BASEDIR"])
            / FargopyConfiguration()["FP_FARGO3D_PACKDIR"]
        }

    @classmethod
    def get_variables(cls) -> dict:
        instance = cls()
        return instance._variables

    @classmethod
    def get(cls, key: str) -> object:
        instance = cls()
        return instance._variables[key]
