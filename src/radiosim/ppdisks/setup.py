import subprocess
from pathlib import Path

from tqdm.auto import tqdm

from radiosim.ppdisks.config.fargo import (
    FargoOptionConfig,
    FargoParameterConfig,
    UnitSystem,
)
from radiosim.ppdisks.config.variables import Variables


class Setup:
    def __init__(
        self,
        name: str,
    ):
        self._name: str = name
        self._path: Path = Variables.get("FARGO_ROOT") / f"setups/{name}"

        if not self._path.exists():
            raise NotADirectoryError("The given setup does not exist.")

        self._option_config: FargoOptionConfig = FargoOptionConfig(
            setup=self._name, autosave=True
        )
        self._param_config: FargoParameterConfig = FargoParameterConfig(
            setup=self._name, autosave=True
        )

    def compile(
        self,
        gpu: bool,
        parallel: bool,
        unit_system: UnitSystem = UnitSystem.MKS,
        rescale: bool = False,
        show_progress: bool = True,
        model_id: int | None = None,
        verbose: bool = False,
        show_fargo_output: bool = False,
    ) -> None:
        model_desc = f" | Model {model_id}" if model_id is not None else ""
        with tqdm(
            desc="Compiling" + model_desc, total=1, disable=not show_progress
        ) as progress:
            if verbose:
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
                stdout=subprocess.DEVNULL if not show_fargo_output else None,
                stderr=subprocess.DEVNULL if not show_fargo_output else None,
                shell=True,
            )

            if verbose:
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
                stdout=subprocess.DEVNULL if not show_fargo_output else None,
                stderr=subprocess.DEVNULL if not show_fargo_output else None,
                shell=True,
            )

            if verbose:
                print("============ FINISH COMPILATION ============")

            progress.update(n=1)

    # Subprocess output capture adapted from https://stackoverflow.com/a/28319191
    # Marked code (inside >>> BEGIN / <<< END) is licensed under CC BY-SA 3.0
    def run(
        self,
        show_progress: bool = True,
        model_id: int | None = None,
        cuda_device_id: int = 0,
    ) -> None:
        total_steps = self._param_config["output_parameters.ntot"].value
        steps_between_outputs = self._param_config["output_parameters.ninterm"].value

        model_desc = f" | Model {model_id}" if model_id is not None else ""

        # >>> BEGIN
        with (
            tqdm(
                desc="Simulating" + model_desc,
                total=total_steps,
                disable=not show_progress,
            ) as progress,
            subprocess.Popen(
                [
                    f"./fargo3d -D {cuda_device_id} "
                    f"setups/{self._name}/{self._param_config._path.name}"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=1,
                universal_newlines=True,
                shell=True,
                cwd=Variables.get("FARGO_ROOT"),
            ) as p,
        ):
            for line in p.stdout:
                if not line.startswith("OUTPUT"):
                    continue

                progress.update(n=steps_between_outputs)

        # <<< END
