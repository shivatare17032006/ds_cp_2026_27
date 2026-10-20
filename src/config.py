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
    def processed_dir(self) -> Path:
        return self.root / self.values["paths"]["processed_dir"]

    @property
    def artifacts_dir(self) -> Path:
        return self.root / self.values["paths"]["artifacts_dir"]

    @property
    def reports_dir(self) -> Path:
        return self.root / self.values["paths"].get("reports_dir", "reports/generated")

    @property
    def dataset_name(self) -> str:
        return str(self.values["project"]["dataset"])

    @property
    def dataset_files(self) -> dict[str, str]:
        return dict(self.values["dataset"][self.dataset_name])

    @property
    def horizon(self) -> int:
        return int(self.values["project"]["horizon_days"])

    @property
    def seed(self) -> int:
        return int(self.values["project"]["seed"])

    @property
    def families(self) -> list[str]:
        """Families in scope; empty means all."""
        return [str(f) for f in self.values.get("scope", {}).get("families") or []]

    @property
    def stores(self) -> list[int]:
        """Stores in scope; empty means all."""
        return [int(s) for s in self.values.get("scope", {}).get("stores") or []]

    @property
    def backtest_folds(self) -> int:
        return int(self.values.get("evaluation", {}).get("folds", 3))

    @property
    def models(self) -> list[str]:
        return [str(m) for m in self.values.get("evaluation", {}).get("models", ["lightgbm"])]

    @property
    def lost_margin(self) -> float:
        return float(self.values.get("assumptions", {}).get("lost_margin", 1.0))

    @property
    def waste_cost(self) -> float:
        return float(self.values.get("assumptions", {}).get("waste_cost", 0.3))

    def shelf_life(self, family: str) -> int:
        """Shelf life in days for ``family``, falling back to ``default``."""
        lives = self.values.get("assumptions", {}).get("shelf_life_days", {})
        return int(lives.get(family, lives.get("default", 7)))


def load_config(path: str | Path) -> PipelineConfig:
    """Load YAML and resolve all project-relative paths from the config folder."""

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    return PipelineConfig(root=config_path.parent.parent, values=values)
