import shutil
import subprocess
from pathlib import Path


def _parse_fpc_value(value: object) -> str:
    match value:
        case str():
            return f'"{value}"'
        case float() | int() | bool():
            return value
        case _:
            raise TypeError(
                f"The value '{value}' is not parsable for the fargopy config!"
            )


class FargopyConfiguration:
    def __init__(self):
        self.path: Path = Path("~/.fargopy/fargopyrc").expanduser()

        if not self.exists():
            self.reset()

    def exists(self) -> bool:
        return self.path.is_file()

    def get_content(self) -> dict:
        fargopyrc = dict()
        with open(self.path) as file:
            exec(file.read(), dict(), fargopyrc)
        return fargopyrc

    def reset(self) -> None:
        try:
            import fargopy as fp

            fp.initialize(options="configure")
        except Exception:
            print(
                "The fargopy configuration seems to be corrupted or is gone. "
                "Re-running initial fargopy configuration. This could take a moment..."
            )

            if self.exists() or self.path.parent.exists():
                shutil.rmtree(self.path.parent)

            subprocess.run(["ifargopy"], shell=True)

            print("Finished regeneration.")

        self["FP_FARGO3D_CLONECMD"] = "git clone https://github.com/FARGO3D/fargo3d.git"

    def __getitem__(self, i: str):
        return self.get_content()[i]

    def __setitem__(self, key: str, value: object):
        _ = self[key]
        with open(self.path) as f:
            content = f.readlines()

        with open(self.path, "w") as f:
            for i in range(len(content)):
                if content[i].startswith(key):
                    content[i] = f"{key} = {_parse_fpc_value(value)}\n"
            f.writelines(content)
