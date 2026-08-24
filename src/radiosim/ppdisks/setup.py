import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from radiosim.ppdisks.config.fargo import (
    FargoOptionConfig,
    FargoParameterConfig,
    UnitSystem,
)
from radiosim.ppdisks.config.variables import Variables

BOUNDARIES_TEMPLATE_URL = "https://raw.githubusercontent.com/FARGO3D/fargo3d/8397dba90088a8e58fb25e5aff639c4e170876fd/setups/fargo_multifluid/boundaries.txt"
BOUND_TEMPLATE_URL = "https://raw.githubusercontent.com/FARGO3D/fargo3d/8397dba90088a8e58fb25e5aff639c4e170876fd/setups/fargo_multifluid/fargo_multifluid.bound.0"
CONDINIT_TEMPLATE_URL = "https://raw.githubusercontent.com/FARGO3D/fargo3d/8397dba90088a8e58fb25e5aff639c4e170876fd/setups/fargo_multifluid/condinit.c"


def _get_dl_command() -> str:
    command = None
    try:
        subprocess.run(
            "curl --version",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        command = "curl"
    except subprocess.CalledProcessError:
        try:
            subprocess.run(
                "wget --version",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            command = "wget"
        except subprocess.CalledProcessError:
            raise OSError(
                "Neither curl nor wget are installed. "
                "Cannot download templates. "
                "Install either of the commands or create the template manually."
            ) from None
    return command


def _dl_file(url: str, command: str, cwd: Path):
    file_path = cwd / url.split("/")[-1]
    match command:
        case "curl":
            subprocess.run(
                f"curl {url} > {file_path.name}",
                shell=True,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=cwd,
            )
        case "wget":
            subprocess.run(
                f"wget {url}",
                shell=True,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=cwd,
            )

    return file_path


class Setup:
    def __init__(self, name: str, create_if_not_exist: bool = True):
        self._name: str = name

        if not Variables.get("FARGO_ROOT").exists():
            raise NotADirectoryError(
                "The FARGO3D directory at "
                f"{Variables.get('FARGO_ROOT')} does not exist. "
                "Install FARGO3D at this location first or change "
                f"the directory at {Path.home() / '.radiosim/config.toml'}."
            )

        self._path: Path = Variables.get("FARGO_ROOT") / f"setups/{name}"

        did_exist = True
        if not self.exists():
            if not create_if_not_exist:
                raise NotADirectoryError("The given setup does not exist.")
            else:
                self._create()
                did_exist = False

        self._option_config: FargoOptionConfig = FargoOptionConfig(
            setup=self._name, autosave=True
        )
        self._param_config: FargoParameterConfig = FargoParameterConfig(
            setup=self._name, autosave=True
        )

        # Update number of bound files and condinit.c based on num of species
        if not did_exist:
            num_species = 0
            for key in self._param_config["dust_parameters"]:
                if key.startswith("invstokes"):
                    num_species += 1

            self.set_num_species(num_species=num_species)
            self._option_config["fluids.NFLUIDS"] = num_species + 1
            self._option_config.save()

    def exists(self) -> bool:
        return self._path.exists()

    def _create(self) -> None:
        self._path.mkdir(exist_ok=True)

        # Download template files
        command = _get_dl_command()
        for file in tqdm(
            [BOUNDARIES_TEMPLATE_URL, BOUND_TEMPLATE_URL, CONDINIT_TEMPLATE_URL],
            desc="Downloading templates",
        ):
            file_path = _dl_file(url=file, command=command, cwd=self._path)

            if file == BOUND_TEMPLATE_URL:
                file_path.rename(file_path.with_stem(f"{self._name}.bound"))

        # Create empty units file
        (self._path / f"{self._name}.units").touch()

        # Create disclaimer file
        (
            self._path / "!WARNING! SETUP MANAGED BY RADIOSIM DO NOT EDIT MANUALLY!"
        ).touch()

        print(f"Created default setup @ {self._path}")

    def set_num_species(self, num_species: int) -> None:
        if num_species < 1:
            raise ValueError("There has to be at least one dust species!")

        # Create corresponding number of bound files

        if not (self._path / f"{self._name}.bound.0").exists():
            command = _get_dl_command()
            file_path = _dl_file(
                url=BOUND_TEMPLATE_URL, command=command, cwd=self._path
            )
            file_path.rename(file_path.with_stem(f"{self._name}.bound"))

        files = [file for file in self._path.glob("*.bound.*")]
        fluid_ids = np.array([int(file.suffix[1:]) for file in files])

        template_file = self._path / f"{self._name}.bound.0"
        for fluid_id in range(1, num_species + 1):
            shutil.copy(
                template_file, template_file.parent / f"{template_file.stem}.{fluid_id}"
            )

        for fluid_id in fluid_ids[fluid_ids > num_species]:
            (self._path / f"{self._name}.bound.{fluid_id}").unlink()

        # Edit condinit.c

        with open(self._path / "condinit.c") as f:
            lines = f.readlines()

        def dustlines():
            dust_lines = {}
            for i in range(len(lines)):
                if lines[i].strip().startswith("ColRate(INVSTOKES"):
                    dust_id = int(
                        lines[i].strip().removeprefix("ColRate(INVSTOKES").split(",")[0]
                    )
                    dust_lines[dust_id] = i
            return dust_lines

        num_spec = 2

        present_dust_idx = np.array(list(dustlines().keys()))

        for del_idx in present_dust_idx[present_dust_idx > num_spec]:
            lines.pop(dustlines()[del_idx])

        present_dust_idx = present_dust_idx[present_dust_idx <= num_spec]

        for dust_idx in range(1, num_spec + 1):
            if dust_idx not in present_dust_idx:
                dust_lines = dustlines()
                lines.insert(
                    dust_lines[dust_idx - 1] + 1,
                    f"  ColRate(INVSTOKES{dust_idx}, id_gas, {dust_idx}, feedback);\n",
                )

        with open(self._path / "condinit.c", "w") as f:
            f.writelines(lines)

    def compile(
        self,
        gpu: bool,
        parallel: bool,
        unit_system: UnitSystem = UnitSystem.MKS,
        rescale: bool = False,
        show_progress: bool = True,
        model_id: int | None = None,
        return_execution_time: bool = False,
        verbose: bool = False,
        show_fargo_output: bool = False,
    ) -> int | None:
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

            clean_cmd = ["make mrproper"]

            if verbose:
                print(f"CMD @ {Variables.get('FARGO_ROOT')}  $ " + " ".join(clean_cmd))

            starting_time = time.time_ns()

            subprocess.run(
                clean_cmd,
                cwd=Variables.get("FARGO_ROOT"),
                stdout=subprocess.DEVNULL if not show_fargo_output else None,
                stderr=subprocess.DEVNULL if not show_fargo_output else None,
                shell=True,
                check=False,
            )

            if verbose:
                print("Starting compilation ...")

            run_cmd = [
                f"make SETUP={self._name} "
                f"GPU={1 if gpu else 0} "
                f"PARALLEL={1 if parallel else 0} "
                f"UNITS={unit_system.key} "
                f"RESCALE={1 if rescale else 0}"
            ]
            if verbose:
                print(f"CMD @ {Variables.get('FARGO_ROOT')}  $ " + " ".join(run_cmd))

            subprocess.run(
                run_cmd,
                cwd=Variables.get("FARGO_ROOT"),
                stdout=subprocess.DEVNULL if not show_fargo_output else None,
                stderr=subprocess.DEVNULL if not show_fargo_output else None,
                shell=True,
                check=False,
            )

            compile_time = time.time_ns() - starting_time

            if verbose:
                print("============ FINISH COMPILATION ============")

            progress.update(n=1)

            if return_execution_time:
                return compile_time
            else:
                return None

    # Subprocess output capture adapted from https://stackoverflow.com/a/28319191
    # Marked code (inside >>> BEGIN / <<< END) is licensed under CC BY-SA 3.0
    def run(
        self,
        show_progress: bool = True,
        model_id: int | None = None,
        parallel: bool = False,
        num_nodes: int = 1,
        cuda_device_id: int = 0,
        return_execution_time: bool = False,
        verbose: bool = False,
    ) -> tuple[float, list[float]] | None:
        total_steps = self._param_config["output_parameters.ntot"].value
        steps_between_outputs = self._param_config["output_parameters.ninterm"].value

        model_desc = f" | Model {model_id}" if model_id is not None else ""

        processes = []

        if parallel:
            processes.append(f"mpirun -np {num_nodes} ")

        processes.extend(
            [
                f"./fargo3d -D {cuda_device_id} "
                f"setups/{self._name}/{self._param_config._path.name}"
            ]
        )

        if verbose:
            print(f"CMD @ {Variables.get('FARGO_ROOT')}  $ " + " ".join(processes))

        starting_time = time.time_ns()
        output_times = []

        # >>> BEGIN
        with (
            tqdm(
                desc="Simulating" + model_desc,
                total=total_steps,
                disable=not show_progress,
            ) as progress,
            subprocess.Popen(
                processes,
                stdout=subprocess.PIPE,
                stderr=None if verbose else subprocess.DEVNULL,
                bufsize=1,
                universal_newlines=True,
                shell=True,
                cwd=Variables.get("FARGO_ROOT"),
            ) as p,
        ):
            for line in p.stdout:
                if not line.startswith("OUTPUT"):
                    continue

                output_times.append(time.time_ns() - starting_time)
                progress.update(n=steps_between_outputs)
        # <<< END

        run_time = time.time_ns() - starting_time

        if return_execution_time:
            return run_time, output_times
        else:
            return None
