"""Small shared contracts. Images are uint8 BGR; masks are boolean crop coordinates."""
from dataclasses import dataclass, field
from typing import Any
import numpy as np

@dataclass
class SpatialCalibration:
    pixels_per_mm: float
    mm_per_pixel: float
    bbox: tuple[int, int, int, int]
    residual: float
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class ColorCalibration:
    matrix: np.ndarray
    bbox: tuple[int, int, int, int]
    mean_delta_e00_before: float
    mean_delta_e00_after: float
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class DishROI:
    dish_id: int
    center: tuple[float, float]
    radius: float
    bbox: tuple[int, int, int, int]
    ellipse: tuple = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    # bbox is x,y,width,height; crop origin is bbox[:2]; no resize.

@dataclass
class SeedInstance:
    seed_id: int
    mask: np.ndarray
    confidence: float
    bbox: tuple[int, int, int, int]
    polygon: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class SeedTraits:
    image_id: str
    dish_id: int
    seed_id: int
    length_mm: float
    width_mm: float
    area_mm2: float
    L: float
    a: float
    b: float
    seg_confidence: float
    qc_flag: str
