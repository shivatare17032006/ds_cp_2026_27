"""Typed configuration loading for the forecasting pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PipelineConfig:
    """Resolved project configuration with paths rooted at the repository."""

    root: Path
    values: dict[str, Any]

    @property
    def raw_dir(self) -> Path:
        return self.root / self.values["paths"]["raw_dir"]

    @property
    def interim_dir(self) -> Path:
        return self.root / self.values["paths"]["interim_dir"]

    @property
    def dataset_name(self) -> str:
        return str(self.values["project"]["dataset"])

    @property
    def dataset_files(self) -> dict[str, str]:
        return dict(self.values["dataset"][self.dataset_name])


def load_config(path: str | Path) -> PipelineConfig:
    """Load YAML and resolve all project-relative paths from the config folder."""

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    return PipelineConfig(root=config_path.parent.parent, values=values)
