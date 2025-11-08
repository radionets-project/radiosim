from pydantic import BaseModel, Field, ValidationInfo, field_validator


__all__ = ["PPDataSetConfig", "PPMetdaData", "PPGeneralConfig"]


class PPMetdaData(BaseModel):
    img_size: int = Field(default_value=512, ge=64)
    alpha_range: list[float] = [0.0, 180.0]
    ratio_range: list[float] = [3.0, 15.0]
    size_ratio_range: list[float] = [0.1, 1.0]
    seed: int | str | None = 1337


class PPDataSetConfig(BaseModel):
    file_prefix: str = "ppdisks"
    batch_size: int = Field(default_value=5, gt=0)
    batches: int = Field(default_value=10, gt=0)
    batches_train_ratio: float = Field(default_value=0.7, gt=0)
    batches_valid_ratio: float = Field(default_value=0.2, gt=0)
    batches_test_ratio: float = Field(default_value=0.1, gt=0)
