import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Type, TypeVar, Union

import yaml
from pydantic import BaseModel

logger = logging.getLogger(f"base.{__file__}")

T = TypeVar("T", bound="BaseConfig")


class BaseConfig(BaseModel):
    """An experiment configuration.

    Inputs are validated using pydantic, to fail early on invalid setup.
    """

    @classmethod
    def from_yaml(cls: Type[T], path: Union[Path, str]) -> T:
        """Load a configuration from a yaml file and instanciate an `ExpConfig`."""

        with Path(path).open("r") as f:
            d = yaml.safe_load(f)

        return cls(**d)

    def to_dict(self) -> Dict:
        """Return a `dict` of the parameters."""
        return deepcopy(vars(self))

    def to_yaml(self, path: Union[Path, str]) -> None:
        """Save a configuration to disk as yaml."""

        with Path(path).open("w") as f:
            yaml.safe_dump(self.to_dict(), f, indent=4)


class DatasetCreationConfig(BaseConfig):
    """An dataset creation configuration."""

    # DEBUG
    source: str = "items.parquet"

    # TODO: add the full query

    # Dataset configuration
    # min_images_listing: int = 4
    # max_images_listing: int = 14
    shard_size: int = 16_384
    # query_created_after: str = "2019-07-01 00:00:00"
    timestamp: Optional[str] = None
    # query_rand_threshold: float = 0.2
    percent_validation: float = 0.0

    # Specific to "filtered_category_brand" dataset
    # min_samples_per_class: int = 512
    # max_samples_per_class: int = 4096
    # total_classes: int = 1024

    # Local and upload directories naming
    save_tar_dir: str = "/tmp/wds/"

    # Set automatically after sharding is computed
    listings_count: Optional[int] = None
    train_shards_count: Optional[int] = None
    val_shards_count: Optional[int] = None

    # GCP
    gcp_project: str = "PROJECT_NAME"
    gcp_bucket: str = "BUCKET_NAME"

    # @validator("min_images_listing", "query_rand_threshold")
    # def more_than_zero(cls, v):
    #     assert v > 0, f"Should be more than 0: {v}"
    #     return v
    #
    # @validator("query_created_after")
    # def valid_date(cls, v):
    #     """Use pandas to flexibily validate dates."""
    #     try:
    #         _ = pd.Timestamp(v)
    #     except Exception as e:
    #         raise ValueError(f"Could not parse the date: {v}") from e
    #     return v
