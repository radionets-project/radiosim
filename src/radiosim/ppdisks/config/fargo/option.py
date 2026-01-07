from dataclasses import dataclass
from enum import Enum

__all__ = [
    "FargoOptionConfig",
    "FargoOptionEntry",
]


class OptionType(Enum):
    VARIABLE = 1
    LIST = 2
    OPTION = 3


@dataclass
class FargoOptionEntry:
    value: object | None
    option_type: OptionType
    enabled: bool = True
    cuda_only: bool = False

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    @classmethod
    def option(
        enabled: bool, value: object | None = None, cuda_only: bool = False
    ) -> "FargoOptionEntry":
        return FargoOptionEntry(
            value=value,
            option_type=OptionType.OPTION,
            enabled=enabled,
            cuda_only=cuda_only,
        )


class FargoOptionConfig:
    def __init__(self):
        self._parameters = {
            "fluids": {
                # This parameter automatically implies the list variable FLUIDS
                # and the option NFLUIDS
                "NFLUIDS": FargoOptionEntry(value="2", option_type=OptionType.VALUE),
                "DRAGFORCE": FargoOptionEntry.option(enabled=True),
                "STOKESNUMBER": FargoOptionEntry.option(enabled=True),
                "DUSTDIFFUSION": FargoOptionEntry.option(enabled=True),
                "COLLISIONPREDICTOR": FargoOptionEntry.option(enabled=False),
            },
            "performance": {"FLOAT": FargoOptionEntry.option(enabled=True)},
            "dimensions": {
                "X": FargoOptionEntry.option(enabled=True),
                "Y": FargoOptionEntry.option(enabled=True),
                "Z": FargoOptionEntry.option(enabled=False),
            },
            "coordinates": {
                "CARTESIAN": FargoOptionEntry.option(enabled=False),
                "CYLINDRICAL": FargoOptionEntry.option(enabled=True),
                "SPHERICAL": FargoOptionEntry.option(enabled=False),
            },
            "planetary_system": {
                "NODEFAULTSTAR": FargoOptionEntry.option(enabled=False),
            },
            "equation_of_state": {
                "ADIABATIC": FargoOptionEntry.option(enabled=False),
                "ISOTHERMAL": FargoOptionEntry.option(enabled=True),
            },
            "additional_physics": {
                "MHD": FargoOptionEntry.option(enabled=False),
                "STRICTSYM": FargoOptionEntry.option(enabled=False),
                "OHMICDIFFUSION": FargoOptionEntry.option(enabled=False),
                "AMBIPOLARDIFFUSION": FargoOptionEntry.option(enabled=False),
                "HALLEFFECT": FargoOptionEntry.option(enabled=False),
                "VISCOSITY": FargoOptionEntry.option(enabled=False),
                "ALPHAVISCOSITY": FargoOptionEntry.option(enabled=True),
                "POTENTIAL": FargoOptionEntry.option(enabled=True),
                "STOCKHOLM": FargoOptionEntry.option(enabled=True),
                "HILLCUT": FargoOptionEntry.option(enabled=False),
            },
            "shearing_box": {
                "SHEARINGBOX": FargoOptionEntry.option(enabled=False),
                "SHEARINGBC": FargoOptionEntry.option(enabled=False),
            },
            "transport": {
                "RAM": FargoOptionEntry.option(enabled=False),
                "STANDARD": FargoOptionEntry.option(enabled=False),
            },
            "slopes": {
                "DONOR": FargoOptionEntry.option(enabled=False),
            },
            "artificial_viscosity": {
                "NOSUBSTEP2": FargoOptionEntry.option(enabled=False),
                "STRONG_SHOCK": FargoOptionEntry.option(enabled=False),
            },
            "boundaries": {
                "HARDBOUNDARIES": FargoOptionEntry.option(enabled=False),
            },
            "utils": {
                "LONGSUMMARY": FargoOptionEntry.option(enabled=True),
            },
            "cuda_blocks": {
                "BLOCK_X": FargoOptionEntry.option(
                    value=16, enabled=True, cuda_only=True
                ),
                "BLOCK_Y": FargoOptionEntry.option(
                    value=16, enabled=True, cuda_only=True
                ),
                "BLOCK_Z": FargoOptionEntry.option(
                    value=1, enabled=True, cuda_only=True
                ),
            },
        }
