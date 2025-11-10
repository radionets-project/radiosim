from decimal import Decimal, getcontext
from typing import Self

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

__all__ = ["PPDataSetConfig", "PPMetdaDataConfig"]


class PPMetdaDataConfig(BaseModel):
    alpha_range: list[float] = [0.0, 180.0]
    ratio_range: list[float] = [3.0, 15.0]
    size_ratio_range: list[float] = [0.1, 1.0]

    @field_validator("alpha_range", "ratio_range", "size_ratio_range")
    @classmethod
    def validate_list_len(cls, v: list[int], info: ValidationInfo) -> list[int]:
        if len(v) != 2:
            raise ValueError(f"Expected '{info.field_name}' to be of length 2!")

        return v


class PPDataSetConfig(BaseModel):
    file_prefix: str = "ppdisks"
    batch_size: int = Field(default_value=5, gt=0)
    batches: int = Field(default_value=10, gt=0)
    batches_train_ratio: float = Field(default_value=0.7, gt=0)
    batches_valid_ratio: float = Field(default_value=0.2, gt=0)
    batches_test_ratio: float = Field(default_value=0.1, gt=0)

    @model_validator(mode="after")
    def validate_data_split(self) -> Self:
        getcontext().prec = 10

        split_ratio_sum = (
            Decimal(self.batches_train_ratio)
            + Decimal(self.batches_valid_ratio)
            + Decimal(self.batches_test_ratio)
        )

        if split_ratio_sum != 1:
            raise ValueError(
                "Expected the sum of data split ratios to be 1 but got "
                f"{split_ratio_sum}. Please make sure your splits are valid."
            )

        return self
