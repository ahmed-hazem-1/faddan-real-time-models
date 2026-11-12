"""
app/config.py
YAML-based crop configuration loader with Pydantic validation.
Schema matches configs/crops/*.yaml exactly.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Sub-models — schema matches the YAML structure exactly
# ---------------------------------------------------------------------------

class ClassDefinition(BaseModel):
    id: int
    name: str
    ignore_in_metrics: bool = False


class NormalizationConfig(BaseModel):
    mean: List[float]
    std: List[float]
    channel_order: str = "rgb_after_bgr_flip"

    @field_validator("mean", "std")
    @classmethod
    def _check_len3(cls, v: List[float]) -> List[float]:
        if len(v) != 3:
            raise ValueError("normalization mean/std must have exactly 3 values (RGB)")
        return v


class ModelSection(BaseModel):
    id: str
    architecture: str                    # e.g. "segformer_b0"
    weights: str                         # relative path, e.g. "weights/stageB_segformer_b0_best.pth"
    input_size: int = 512                # square input side length
    normalization: NormalizationConfig


class ClaheConfig(BaseModel):
    clip: float = 2.0
    tiles: List[int] = Field(default=[8, 8], min_length=2, max_length=2)


class GlareConfig(BaseModel):
    pctl: float = 99.0
    min_L: float = 240.0
    dilate: int = 7
    radius: int = 3
    max_frac: float = 0.20


class BilateralConfig(BaseModel):
    d: int = 5
    sigma: float = 50.0


class UnsharpConfig(BaseModel):
    amount: float = 0.5
    radius: int = 1


class BrightnessConfig(BaseModel):
    lo: float = 110.0
    hi: float = 140.0
    gain_clip: List[float] = Field(default=[0.6, 1.5], min_length=2, max_length=2)


class PreprocessingSection(BaseModel):
    wb_gain_clip: List[float] = Field(default=[0.5, 2.0], min_length=2, max_length=2)
    clahe: ClaheConfig = Field(default_factory=ClaheConfig)
    glare: GlareConfig = Field(default_factory=GlareConfig)
    bilateral: BilateralConfig = Field(default_factory=BilateralConfig)
    unsharp: UnsharpConfig = Field(default_factory=UnsharpConfig)
    brightness: BrightnessConfig = Field(default_factory=BrightnessConfig)


class SeveritySection(BaseModel):
    early_max: float = 0.05
    mid_max: float = 0.20

    def bucket(self, ratio: float) -> str:
        if ratio < self.early_max:
            return "early"
        if ratio < self.mid_max:
            return "mid"
        return "late"


class InferenceSection(BaseModel):
    mode: str = "whole_image"
    input_size: int = 512


class CleanupSection(BaseModel):
    morph_open: bool = False
    morph_open_ksize: int = 3
    min_component_px: int = 0  # 0 = disabled


class TilingSection(BaseModel):
    enabled: bool = False
    size: int = 512
    stride: int = 448


# ---------------------------------------------------------------------------
# Root config model
# ---------------------------------------------------------------------------

class CropConfig(BaseModel):
    crop: str
    model: ModelSection
    classes: List[ClassDefinition]
    preprocessing: PreprocessingSection = Field(default_factory=PreprocessingSection)
    severity: SeveritySection = Field(default_factory=SeveritySection)
    inference: InferenceSection = Field(default_factory=InferenceSection)
    cleanup: CleanupSection = Field(default_factory=CleanupSection)
    tiling: TilingSection = Field(default_factory=TilingSection)

    # ---- validators --------------------------------------------------------

    @field_validator("crop")
    @classmethod
    def _crop_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("'crop' must not be empty")
        return v.strip()

    @field_validator("classes")
    @classmethod
    def _classes_nonempty(cls, v: List[ClassDefinition]) -> List[ClassDefinition]:
        if not v:
            raise ValueError("'classes' list must contain at least one entry")
        ids = [c.id for c in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate class ids found in 'classes'")
        return v

    # ---- helpers -----------------------------------------------------------

    def get_disease_classes(self) -> List[ClassDefinition]:
        """Return non-background classes (ignore_in_metrics=False)."""
        return [c for c in self.classes if not c.ignore_in_metrics]

    def class_name_map(self) -> Dict[int, str]:
        """Return {class_id: snake_case_name} for all classes."""
        def _snake(name: str) -> str:
            return re.sub(r"[\s\-]+", "_", name).lower()
        return {c.id: _snake(c.name) for c in self.classes}

    def num_classes(self) -> int:
        return len(self.classes)

    def weights_path(self, project_root: Optional[Path] = None) -> Path:
        p = Path(self.model.weights)
        if p.is_absolute():
            return p
        root = project_root or Path(__file__).resolve().parent.parent
        return root / p

    def input_size(self) -> int:
        return self.model.input_size

    def norm_mean(self) -> List[float]:
        return self.model.normalization.mean

    def norm_std(self) -> List[float]:
        return self.model.normalization.std


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(crop_name: str, config_root: Optional[Path] = None) -> CropConfig:
    """
    Load and validate the YAML config for *crop_name*.

    Config file is resolved as:
        <config_root>/configs/crops/<crop_name>.yaml

    If *config_root* is None, the project root is assumed to be the directory
    two levels above this file (i.e. repo root when this file lives in app/).
    """
    if not crop_name or not crop_name.strip():
        raise ValueError("crop_name must not be empty")

    crop_name = crop_name.strip().lower()

    if config_root is None:
        # Resolve relative to this file: app/config.py -> app/ -> project root
        config_root = Path(__file__).resolve().parent.parent

    yaml_path = config_root / "configs" / "crops" / f"{crop_name}.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Crop config not found: {yaml_path}\n"
            f"Expected a YAML file at configs/crops/{crop_name}.yaml"
        )

    try:
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML at {yaml_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(
            f"Config at {yaml_path} must be a YAML mapping (dict), got {type(raw)}"
        )

    try:
        config = CropConfig(**raw)
    except Exception as exc:
        raise ValueError(
            f"Config validation failed for crop '{crop_name}' ({yaml_path}):\n{exc}"
        ) from exc

    return config
