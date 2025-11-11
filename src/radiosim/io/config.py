import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from ._ppdisks_config import PPDataSetConfig, PPMetdaDataConfig

__all__ = [
    "Config",
    "GeneralConfig",
    "PathConfig",
    "SurveyConfig",
    "JetConfig",
    "MojaveConfig",
    "PPDiskConfig",
    "DataSetConfig",
]


class GeneralConfig(BaseModel):
    verbose: bool = False
    seed: int | str | None = None
    threads: int | str | None = None
    device: str = "cuda"

    @field_validator("seed", "threads")
    @classmethod
    def parse_seed(cls, v: str | bool | int | None) -> int | None:
        if v in {"none", False}:
            v = None

        return v


class PathConfig(BaseModel, validate_assignment=True):
    outpath: str | Path = "./build/example_data/"

    @field_validator("outpath")
    @classmethod
    def expand_path(cls, v: Path, info: ValidationInfo) -> Path:
        """Expand and resolve paths."""

        if v in {None, False, "none", ""}:
            raise ValueError(f"'{info.field_name}' cannot be empty!")
        else:
            v = Path(v)
            v.expanduser().resolve()

        return v


class JetConfig(BaseModel):
    training_type: Literal["list", "gauss", "clean"] = "list"
    num_jet_components: list[int] = [3, 10]
    scaling: Literal["normalize", "mojave"] = "normalize"

    @field_validator("num_jet_components")
    @classmethod
    def validate_list_len(cls, v: list[int], info: ValidationInfo) -> list[int]:
        if len(v) != 2:
            raise ValueError(f"Expected '{info.field_name}' to be of length 2!")

        return v


class SurveyConfig(BaseModel):
    num_sources: int = 20
    class_distribution: list[int] = [2, 1, 2]
    scale_sources: bool = True

    @field_validator("class_distribution")
    @classmethod
    def validate_list_len(cls, v: list[int], info: ValidationInfo) -> list[int]:
        if len(v) != 3:
            raise ValueError(f"Expected '{info.field_name}' to be of length 3!")

        return v


class MojaveConfig(BaseModel, validate_assignment=True):
    class_ratio: list[int] = [1, 1, 1]

    @field_validator("class_ratio")
    @classmethod
    def validate_list_len(cls, v: list[int], info: ValidationInfo) -> list[int]:
        if len(v) != 3:
            raise ValueError(f"Expected '{info.field_name}' to be of length 3!")

        return v


class PPDiskConfig(BaseModel):
    metadata: dict | Callable = PPMetdaDataConfig
    dataset: dict | Callable = PPDataSetConfig

    @field_validator("metadata", mode="after")
    @classmethod
    def validate_metadata(cls, v):
        if isinstance(v, dict):
            return PPMetdaDataConfig(**v)

        return v

    @field_validator("dataset", mode="after")
    @classmethod
    def validate_dataset(cls, v):
        if isinstance(v, dict):
            return PPDataSetConfig(**v)

        return v


class DataSetConfig(BaseModel, validate_assignment=True):
    bundles_train: int = Field(default_value=1, ge=0)
    bundles_valid: int = Field(default_value=1, ge=0)
    bundles_test: int = Field(default_value=1, ge=0)
    bundle_size: int = Field(default_value=100, ge=1)
    img_size: int = Field(default_value=512, ge=64)
    noise: bool = True
    noise_level: list[float] = [0.0, 15.0]

    @field_validator("noise_level")
    @classmethod
    def validate_list_len(cls, v: list[int], info: ValidationInfo) -> list[int]:
        if len(v) != 2:
            raise ValueError(f"Expected '{info.field_name}' to be of length 2!")

        return v


class Config(BaseModel):
    """Main training configuration."""

    general: GeneralConfig = Field(default_factory=GeneralConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    dataset: DataSetConfig = Field(default_factory=DataSetConfig)
    jet: JetConfig = Field(default_factory=JetConfig)
    survey: SurveyConfig = Field(default_factory=SurveyConfig)
    mojave: MojaveConfig = Field(default_factory=MojaveConfig)
    ppdisk: PPDiskConfig = Field(default_factory=PPDiskConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> "Config":
        """Load configuration from a TOML file."""
        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls(**data)

    def to_dict(self) -> dict:
        """Export configuration as a dictionary."""
        return self.model_dump()
