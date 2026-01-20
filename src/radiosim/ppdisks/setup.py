import subprocess
from os import PathLike
from pathlib import Path

from radiosim.ppdisks.config.fargo import (
    FargoOptionConfig,
    FargoParameterConfig,
    UnitSystem,
)
from radiosim.ppdisks.config.toml import TOMLConfiguration
from radiosim.ppdisks.config.variables import Variables


class Setup:
    def __init__(self, name: str, config: PathLike, autosave: bool = True):
        self._name: str = name
        self._autosave: bool = autosave
        self._path: Path = Variables.get("FARGO_ROOT") / f"setups/{name}"
        self._config: TOMLConfiguration = TOMLConfiguration(path=config)
        self._option_config: FargoOptionConfig = FargoOptionConfig(
            setup=self._name, autosave=self._autosave
        )
        self._param_config: FargoParameterConfig = FargoParameterConfig(
            setup=self._name, autosave=self._autosave
        )
        self._planet_config: FargoParameterConfig | None = None

    def exists(self) -> bool:
        return self._path.exists()

    def compile(
        self,
        gpu: bool,
        parallel: bool,
        unit_system: UnitSystem = UnitSystem.MKS,
        rescale: bool = False,
        verbose: bool = False,
    ) -> None:
        print(f"========= COMPILE SETUP '{self._name}' =========")
        print(
            f"Options: GPU={gpu}, "
            f"PARALLEL={parallel}, "
            f"UNITS={unit_system.key}, "
            f"RESCALE={rescale}"
        )

        print("Cleaning up make process ...")
        subprocess.run(
            ["make mrproper"],
            cwd=Variables.get("FARGO_ROOT"),
            stdout=subprocess.DEVNULL if not verbose else None,
            stderr=subprocess.DEVNULL if not verbose else None,
            shell=True,
        )

        print("Starting compilation ...")

        subprocess.run(
            [
                f"make SETUP={self._name} "
                f"GPU={1 if gpu else 0} "
                f"PARALLEL={1 if parallel else 0} "
                f"UNITS={unit_system.key} "
                f"RESCALE={1 if rescale else 0}"
            ],
            cwd=Variables.get("FARGO_ROOT"),
            stdout=subprocess.DEVNULL if not verbose else None,
            stderr=subprocess.DEVNULL if not verbose else None,
            shell=True,
        )

        print("============ FINISH COMPILATION ============")
