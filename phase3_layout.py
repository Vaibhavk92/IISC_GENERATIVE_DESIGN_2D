"""
phase3_layout.py — Phase 3: Greedy Placement Engine

Algorithm (single scale):
  1. Compute the distance transform of the current remaining target polygon
     to find its thickest structural zones (torso, head, limbs…).
  2. Sort inventory pieces largest -> smallest.
  3. For each (zone, piece, rotation) triplet, simulate placing the piece
     centred on the zone.  Score by coverage x efficiency.
  4. Commit the best placement: record cut lines, subtract covered area.
  5. Repeat until inventory is exhausted or remaining area is negligible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import MultiPolygon, Polygon, Point

from geometry_utils import (
    EPSILON,
    extract_cut_lines,
    find_structural_zones,
    polygon_area_safe,
    rotate_polygon,
    safe_difference,
    safe_intersection,
    translate_polygon,
    _as_valid,
    _collect_polygons,
)
from models import EnvironmentBounds, InventoryPiece, Layout, PlacedPiece

logger = logging.getLogger(__name__)

# Minimum coverage thresholds.
# MIN_COVERAGE_RATIO: fraction of remaining-target area that must be covered.
#   Kept small so real-world scrap pieces (physically small relative to a large
#   target) are still placed rather than silently skipped.
MIN_COVERAGE_RATIO = 0.001   # 0.1%  (was 0.03 — too strict for small scraps)
# Absolute floor in world-units²: guards against degenerate near-zero overlaps.
MIN_COVERAGE_ABS = 10.0

# Candidate rotation angles (degrees)
DEFAULT_ROTATIONS = list(range(0, 360, 30))


# ---------------------------------------------------------------------------
# Internal candidate holder (not exposed to callers)
# ---------------------------------------------------------------------------

@dataclass
class _Candidate:
    piece: InventoryPiece
    angle: float
    dx: float
    dy: float
    transformed: Polygon
    intersection: Polygon
    waste: Polygon
    zone_x: float
    zone_y: float

    @property
    def coverage(self) -> float:
        return polygon_area_safe(self.intersection)

    @property
    def waste_area(self) -> float:
        return polygon_area_safe(self.waste)

    @property
    def efficiency(self) -> float:
        """Fraction of the placed piece that is usefully inside the target."""
        total = self.coverage + self.waste_area
        return self.coverage / total if total > 0 else 0.0

    def placement_score(self, remaining_area: float) -> float:
        """
        Composite score: rewards high coverage and low waste.
        Normalised so both terms are on a comparable scale.
        """
        norm_coverage = self.coverage / max(remaining_area, EPSILON)
        return 0.65 * norm_coverage + 0.35 * self.efficiency


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class GreedyPlacementEngine:
    """
    Greedy piece placer for a single scale step.

    Each call to `run()` returns a Layout describing where every piece was
    placed, what was cut, and how much target area remains uncovered.
    """

    def __init__(
        self,
        inventory: List[InventoryPiece],
        target_polygon: Polygon,
        bounds: EnvironmentBounds,
        n_structural_zones: int = 8,
        rotation_candidates: Optional[List[float]] = None,
        min_remaining_area: float = 1.0,
        allow_piece_reuse: bool = False,
    ):
        """
        Parameters
        ----------
        inventory            : All available pieces (will be sorted largest-first).
        target_polygon       : The scaled target shape to fill.
        bounds               : Environment bounding box (for guard-rail clipping).
        n_structural_zones   : How many hotspot zones to probe on each iteration.
        rotation_candidates  : Rotation angles (°) to try per piece per zone.
        min_remaining_area   : Stop when remaining target area falls below this.
        allow_piece_reuse    : If True, the same piece can be placed multiple times.
        """
        self.inventory = sorted(inventory, key=lambda p: p.area, reverse=True)
        self.target_polygon = _as_valid(target_polygon)
        self.bounds = bounds
        self.n_zones = n_structural_zones
        self.rotations = rotation_candidates or DEFAULT_ROTATIONS
        self.min_remaining_area = min_remaining_area
        self.allow_reuse = allow_piece_reuse

    # ------------------------------------------------------------------
    def run(self) -> Layout:
        """Execute the greedy loop and return the final Layout."""
        layout = Layout(
            target_polygon=self.target_polygon,
            remaining_target=_as_valid(self.target_polygon),
        )
        available = list(self.inventory)   # pieces not yet placed (if not reusing)

        iteration = 0
        while available and self._has_coverage_left(layout):
            iteration += 1
            remaining = layout.remaining_target

            logger.debug(
                "Iter %d | remaining_area=%.1f | pieces_left=%d",
                iteration, polygon_area_safe(remaining), len(available),
            )

            zones = find_structural_zones(remaining, n_zones=self.n_zones)
            if not zones:
                logger.info("No structural zones found — stopping.")
                break

            best = self._find_best_placement(available, remaining, zones)
            if best is None:
                logger.info("No valid placement at iteration %d — stopping.", iteration)
                break

            placed_piece = self._commit(best, remaining)
            layout.placed_pieces.append(placed_piece)

            # Subtract the newly covered area from the remaining target
            layout.remaining_target = self._subtract_covered(
                layout.remaining_target, placed_piece.intersection_with_target
            )

            if not self.allow_reuse:
                available = [p for p in available if p.piece_id != best.piece.piece_id]

            logger.debug(
                "  -> Placed '%s' at angle=%.0f°, coverage=%.1f, waste=%.1f, cuts=%d",
                best.piece.piece_id, best.angle, best.coverage,
                best.waste_area, len(placed_piece.cut_lines),
            )

        return layout

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _has_coverage_left(self, layout: Layout) -> bool:
        rem = layout.remaining_target
        return rem is not None and not rem.is_empty and rem.area > self.min_remaining_area

    def _find_best_placement(
        self,
        available: List[InventoryPiece],
        remaining: Polygon,
        zones: List[Tuple[float, float, float]],
    ) -> Optional[_Candidate]:
        """
        Exhaustive search over (piece x zone x rotation).
        Returns the candidate with the highest placement_score.
        """
        best: Optional[_Candidate] = None
        best_score = -1.0
        remaining_area = polygon_area_safe(remaining)

        for piece in available:
            for zone_x, zone_y, zone_thickness in zones:
                for angle in self.rotations:
                    cand = self._try_placement(
                        piece, remaining, zone_x, zone_y, angle
                    )
                    if cand is None:
                        continue

                    # Skip placements that barely cover anything
                    if cand.coverage < max(remaining_area * MIN_COVERAGE_RATIO,
                                           MIN_COVERAGE_ABS):
                        continue

                    score = cand.placement_score(remaining_area)
                    if score > best_score:
                        best_score = score
                        best = cand

        return best

    def _try_placement(
        self,
        piece: InventoryPiece,
        remaining: Polygon,
        zone_x: float,
        zone_y: float,
        angle: float,
    ) -> Optional[_Candidate]:
        """
        Rotate the piece, centre it on (zone_x, zone_y), compute geometry.
        Returns None if the intersection is empty (piece misses the zone).
        """
        try:
            rotated = rotate_polygon(piece.polygon, angle)
            if rotated is None or rotated.is_empty:
                return None

            # Align piece centroid to zone centre
            cx, cy = rotated.centroid.coords[0]
            dx, dy = zone_x - cx, zone_y - cy
            placed = translate_polygon(rotated, dx, dy)
            if placed is None or placed.is_empty:
                return None

            # Clip to environment bounds (pieces must not leave the work area)
            placed = safe_intersection(placed, self.bounds.polygon)
            if placed is None or placed.is_empty:
                return None

            intersection = safe_intersection(placed, remaining)
            if intersection is None or intersection.is_empty:
                return None

            waste = safe_difference(placed, remaining)
            waste_poly = _largest_valid_polygon(waste)

            return _Candidate(
                piece=piece,
                angle=angle,
                dx=dx, dy=dy,
                transformed=placed,
                intersection=intersection,
                waste=waste_poly or Polygon(),
                zone_x=zone_x,
                zone_y=zone_y,
            )

        except Exception as exc:
            logger.debug(
                "_try_placement failed (piece=%s angle=%.0f): %s",
                piece.piece_id, angle, exc,
            )
            return None

    def _commit(self, candidate: _Candidate, remaining: Polygon) -> PlacedPiece:
        """Convert a winning _Candidate into a PlacedPiece with cut lines computed."""
        cut_lines = extract_cut_lines(candidate.transformed, remaining)
        waste = candidate.waste if not candidate.waste.is_empty else None

        return PlacedPiece(
            piece=candidate.piece,
            position=(candidate.dx, candidate.dy),
            rotation_degrees=candidate.angle,
            transformed_polygon=_as_valid(candidate.transformed),
            intersection_with_target=_as_valid(candidate.intersection),
            waste_polygon=_as_valid(waste) if waste else None,
            cut_lines=cut_lines,
        )

    @staticmethod
    def _subtract_covered(
        remaining: Polygon,
        covered: Polygon,
    ):
        """
        Subtract the covered area from remaining target.
        Handles MultiPolygon returns gracefully (multiple disconnected islands).
        """
        result = safe_difference(remaining, covered)
        if result is None or result.is_empty:
            return Polygon()

        # If difference created a MultiPolygon, keep all islands (the engine
        # will find zones within whichever island is thickest next iteration)
        return _as_valid(result)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _largest_valid_polygon(geom) -> Optional[Polygon]:
    """Return the largest valid Polygon from any geometry type."""
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom
    polys = _collect_polygons(geom)
    return max(polys, key=lambda p: p.area) if polys else None
