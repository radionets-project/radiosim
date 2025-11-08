import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ._ppdisks_config import PPDataSetConfig


class GeneralConfig(BaseModel):
    quiet: bool = True
    seed: int | str | None = None
    threads: int | str | None = None


class PathConfig(BaseModel):
    outpath: str | Path = "./build/example_data/"


class JetConfig(BaseModel):
    training_type: Literal["list", "gauss", "clean"] = "list"
    num_jet_components: list[int] = [3, 10]
    scaling: Literal["normalize", "mojave"] = "normalize"


class SurveyConfig(BaseModel):
    num_sources: int = 20
    class_distribution: list[int] = [2, 1, 2]
    scale_sources: bool = True


class MojaveConfig(BaseModel):
    class_ratio: list[int] = [1, 1, 1]


class PPDiskConfig(BaseModel):
    metadata: PPMetdaDataConfig = PPMetdaDataConfig
    dataset: PPDataSetConfig = PPDataSetConfig


class DataSetConfig(BaseModel):
    bundles_train: int = Field(default_value=1, ge=0)
    bundles_valid: int = Field(default_value=1, ge=0)
    bundles_test: int = Field(default_value=1, ge=0)
    bundle_size: int = Field(default_value=100, ge=1)
    img_size: int = Field(default_value=256, ge=32)
    noise: bool = True
    noise_level: list[float] = [0.0, 15.0]


class Config(BaseModel):
    """Main training configuration."""

    general: GeneralConfig = Field(default_factory=GeneralConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    jet: JetConfig = Field(default_factory=JetConfig)
    survey: SurveyConfig = Field(default_factory=SurveyConfig)
    mojave: MojaveConfig = Field(default_factory=MojaveConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> "Config":
        """Load configuration from a TOML file."""
        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls(**data)

    def to_dict(self) -> dict:
        """Export configuration as a dictionary."""
        return self.model_dump()
