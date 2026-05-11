import shutil
import subprocess
from os import PathLike
from pathlib import Path

from .config.meta import MetaConfig
from .config.variables import Variables


def install_fargo(
    root_dir: PathLike | str,
    use_ssh: bool = False,
    repository: str = "FARGO3D/fargo3d",
    remote_url: str = "github.com",
    reinstall: bool = False,
    clean_install: bool = False,
) -> None:
    try:
        fargo_dir = Variables.get("FARGO_ROOT")
        if not reinstall:
            raise IsADirectoryError(
                "FARGO3D is already installed. To reinstall, set reinstall=True!"
            )
        if clean_install:
            print("Cleaning previous FARGO3D installation ...")
            shutil.rmtree(fargo_dir)
    except KeyError:
        pass

    root_dir = Path(root_dir)
    root_dir.mkdir(exist_ok=True, parents=True)

    repository_url = (
        f"git@{remote_url}:{repository}.git"
        if use_ssh
        else f"https://github.com/{repository}.git"
    )

    print("Cloning FARGO3D repository ...")

    subprocess.run(f"git clone {repository_url}", shell=True, cwd=root_dir)
    MetaConfig().get_config()["FARGO_ROOT"] = str(
        (root_dir / repository.split("/")[1]).absolute()
    )

    print("FARGO3D was successfully installed!")


def install_radmc3d(
    root_dir: PathLike | str,
    use_ssh: bool = False,
    repository: str = "dullemond/radmc3d-2.0",
    remote_url: str = "github.com",
    reinstall: bool = False,
    ignore_requirements: bool = False,
    skip_compilation: bool = False,
    clean_install: bool = False,
) -> None:
    if not ignore_requirements:
        try:
            subprocess.run(
                "gfortran --version",
                check=True,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            print(
                "gfortran (GNU Fortran) is not installed on your system. "
                "Install it before installing RADMC3D.\n\n"
                "If you want to use a different Fortran Compiler, you have to "
                "set ignore_requirements=True and skip_compilation=True and then "
                "set the compiler manually as described in the RADMC3D manual "
                "in the section 'Installation'."
            )

    try:
        radmc3d_dir = Variables.get("RADMC3D_ROOT")
        if not reinstall:
            raise IsADirectoryError(
                "RADMC3D is already installed. To reinstall, set reinstall=True!"
            )
        if clean_install:
            print("Cleaning previous RADMC3D installation ...")
            shutil.rmtree(radmc3d_dir)
    except KeyError:
        pass

    root_dir = Path(root_dir)
    root_dir.mkdir(exist_ok=True, parents=True)

    repository_url = (
        f"git@{remote_url}:{repository}.git"
        if use_ssh
        else f"https://{remote_url}/{repository}.git"
    )

    print("Cloning RADMC3D repository ...")

    subprocess.run(f"git clone {repository_url}", shell=True, cwd=root_dir)
    radmc_dir = root_dir / repository.split("/")[1]
    MetaConfig().get_config()["RADMC3D_ROOT"] = str(radmc_dir.absolute())

    if not skip_compilation:
        print("Compiling RADMC3D ...")
        subprocess.run("make", shell=True, cwd=radmc_dir / "src")

    print("RADMC3D was successfully installed!")
