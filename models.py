"""
models.py — Core data models for the Generative Design 2D pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString


@dataclass
class InventoryPiece:
    """A single physical piece in the material inventory."""
    piece_id: str
    polygon: Polygon
    material: str = "default"

    @property
    def area(self) -> float:
        return self.polygon.area

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return self.polygon.bounds

    def __repr__(self) -> str:
        return f"InventoryPiece(id={self.piece_id}, area={self.area:.1f})"


@dataclass
class CutLine:
    """Represents a single cut made through a physical piece."""
    geometry: LineString

    @property
    def length(self) -> float:
        return self.geometry.length

    @property
    def is_straight(self) -> bool:
        """True if the cut is a single straight segment."""
        return len(list(self.geometry.coords)) <= 2

    @property
    def complexity(self) -> float:
        """Higher = more complex cut (more vertices, longer path)."""
        n_pts = len(list(self.geometry.coords))
        return self.length * (1.0 + 0.1 * max(0, n_pts - 2))


@dataclass
class PlacedPiece:
    """
    A committed placement of one inventory piece.
    Captures the full transformation and cut geometry.
    """
    piece: InventoryPiece
    position: Tuple[float, float]        # (dx, dy) applied after rotation
    rotation_degrees: float
    transformed_polygon: Polygon         # piece after rotate + translate
    intersection_with_target: Polygon    # the actually covered area
    waste_polygon: Optional[Polygon]     # part of piece outside target (cut away)
    cut_lines: List[CutLine] = field(default_factory=list)

    @property
    def covered_area(self) -> float:
        if self.intersection_with_target and not self.intersection_with_target.is_empty:
            return self.intersection_with_target.area
        return 0.0

    @property
    def waste_area(self) -> float:
        if self.waste_polygon and not self.waste_polygon.is_empty:
            return self.waste_polygon.area
        return 0.0

    @property
    def num_cuts(self) -> int:
        return len(self.cut_lines)

    @property
    def total_cut_length(self) -> float:
        return sum(cl.length for cl in self.cut_lines)

    @property
    def total_cut_complexity(self) -> float:
        return sum(cl.complexity for cl in self.cut_lines)


@dataclass
class Layout:
    """Complete layout result: all placed pieces plus residual geometry."""
    placed_pieces: List[PlacedPiece] = field(default_factory=list)
    remaining_target: Optional[Polygon] = None   # uncovered area remaining
    target_polygon: Optional[Polygon] = None     # original full target

    @property
    def total_covered_area(self) -> float:
        return sum(p.covered_area for p in self.placed_pieces)

    @property
    def uncovered_area(self) -> float:
        if self.remaining_target is None or self.remaining_target.is_empty:
            return 0.0
        return self.remaining_target.area

    @property
    def total_waste_area(self) -> float:
        return sum(p.waste_area for p in self.placed_pieces)

    @property
    def total_cuts(self) -> int:
        return sum(p.num_cuts for p in self.placed_pieces)

    @property
    def total_cut_length(self) -> float:
        return sum(p.total_cut_length for p in self.placed_pieces)

    @property
    def num_pieces_used(self) -> int:
        return len(self.placed_pieces)

    @property
    def coverage_ratio(self) -> float:
        if self.target_polygon is None or self.target_polygon.area == 0:
            return 0.0
        return self.total_covered_area / self.target_polygon.area


@dataclass
class FitnessResult:
    """Outcome of evaluating one scale step."""
    scale_factor: float
    score: float
    layout: Layout
    weights: Tuple[float, float, float, float]   # (w1, w2, w3, w4)

    def __lt__(self, other: FitnessResult) -> bool:
        return self.score < other.score


@dataclass
class EnvironmentBounds:
    """Maximum bounding box that the final output must fit within."""
    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def polygon(self) -> Polygon:
        return Polygon([
            (0, 0), (self.width, 0),
            (self.width, self.height), (0, self.height),
        ])
