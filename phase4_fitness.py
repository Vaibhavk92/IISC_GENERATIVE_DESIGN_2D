"""
phase4_fitness.py — Phase 4: Fitness Scoring & Best Layout Selection

Fitness function (higher = better):

    Score = - w1 * cut_quality_norm          (count + shape complexity of cuts)
            - w2 * fragmentation_norm        (scattered uncovered holes vs one gap)
            - w3 * coverage_penalty          (non-linear: punishes large gaps harder)
            - w4 * material_waste_norm       (waste / piece area, not target area)

Scale is varied externally (50 → 100 %) and is NOT part of the formula.
All penalty terms are normalised to [0, 1].
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from geometry_utils import _collect_polygons
from models import FitnessResult, Layout

logger = logging.getLogger(__name__)

# Upper bounds for normalisation
_MAX_CUTS = 30
_COMPLEXITY_PERIM_RATIO = 3.0   # max expected total_cut_complexity / target_perimeter
_MAX_HOLE_COMPONENTS = 8        # more fragmented than this → max penalty


class FitnessEvaluator:
    """
    Computes a scalar fitness score for a Layout.

    Improvements over v1
    --------------------
    1. **Non-linear coverage penalty** (``uncov_norm ** 1.2``):
       Getting from 90 % to 95 % coverage is penalised more than 50 % → 55 %,
       which better reflects the real difficulty of filling the last gaps.

    2. **Cut quality** replaces raw cut count:
       Combines normalised cut count (0.5) with normalised cut *complexity*
       (0.5), where complexity = length × (1 + 0.1 × extra_vertices).
       A curved cut is costlier than a straight one of the same length.

    3. **Fragmentation penalty** replaces raw cut length:
       Counts the number of disconnected uncovered components.
       Many scattered holes are worse than one large gap because each hole
       would need its own patch piece.  Weighted by uncov_norm so it only
       matters when there IS uncovered area.

    4. **Material efficiency** replaces waste / target_area:
       Measures waste / total_piece_area instead.
       A piece where 80 % is cut away is inefficient regardless of target size.

    Parameters
    ----------
    weights : (w1, w2, w3, w4)
        Default (0.05, 0.04, 0.85, 0.06) — coverage still dominates but the
        other terms are more meaningfully balanced.
    """

    def __init__(
        self,
        weights: Tuple[float, float, float, float] = (0.05, 0.04, 0.85, 0.06),
    ):
        self.w1, self.w2, self.w3, self.w4 = weights

    # ------------------------------------------------------------------
    def score(self, layout: Layout) -> float:
        """Compute and return the scalar fitness score."""
        if layout.target_polygon is None or layout.target_polygon.is_empty:
            return 0.0

        target_area = layout.target_polygon.area
        if target_area == 0:
            return 0.0
        target_perim = layout.target_polygon.exterior.length

        # ── Penalty 1: cut quality (count + complexity) ───────────────
        cuts_norm = min(layout.total_cuts / _MAX_CUTS, 1.0)

        total_complexity = sum(
            cl.complexity
            for pp in layout.placed_pieces
            for cl in pp.cut_lines
        )
        max_complexity = target_perim * _COMPLEXITY_PERIM_RATIO
        complexity_norm = (
            min(total_complexity / max_complexity, 1.0) if max_complexity > 0 else 0.0
        )
        cut_quality_norm = 0.5 * cuts_norm + 0.5 * complexity_norm

        # ── Penalty 2: fragmentation of uncovered area ────────────────
        # More disconnected uncovered holes → harder to patch → higher penalty.
        # Weighted by uncov_norm so it's zero when coverage is perfect.
        uncov_norm = min(layout.uncovered_area / target_area, 1.0)

        if layout.remaining_target and not layout.remaining_target.is_empty:
            n_holes = len(_collect_polygons(layout.remaining_target))
            hole_norm = min(n_holes / _MAX_HOLE_COMPONENTS, 1.0)
        else:
            n_holes = 0
            hole_norm = 0.0

        # Scale fragmentation by how much area is uncovered
        fragmentation_norm = uncov_norm * (0.4 + 0.6 * hole_norm)

        # ── Penalty 3: coverage (non-linear) ──────────────────────────
        # Power > 1 → penalises large gaps disproportionately more.
        coverage_penalty = uncov_norm ** 1.2

        # ── Penalty 4: material efficiency ────────────────────────────
        # Waste relative to how much piece material was actually placed.
        total_piece_area = sum(pp.piece.area for pp in layout.placed_pieces)
        if total_piece_area > 0:
            material_waste_norm = min(layout.total_waste_area / total_piece_area, 1.0)
        elif target_area > 0:
            material_waste_norm = min(layout.total_waste_area / target_area, 1.0)
        else:
            material_waste_norm = 0.0

        fitness = (
            - self.w1 * cut_quality_norm
            - self.w2 * fragmentation_norm
            - self.w3 * coverage_penalty
            - self.w4 * material_waste_norm
        )

        logger.debug(
            "score | cut_quality=%.3fx%.2f | frag=%.3f(%dh)x%.2f | "
            "coverage=%.3f^1.2=%.3fx%.2f | waste_eff=%.3fx%.2f -> %.4f",
            cut_quality_norm, self.w1,
            fragmentation_norm, n_holes, self.w2,
            uncov_norm, coverage_penalty, self.w3,
            material_waste_norm, self.w4,
            fitness,
        )

        return fitness

    # ------------------------------------------------------------------
    def breakdown(self, layout: Layout, scale_factor: float) -> Dict:
        """Return a detailed dict of score components for reporting."""
        if layout.target_polygon is None or layout.target_polygon.is_empty:
            return {}

        target_area = layout.target_polygon.area
        target_perim = layout.target_polygon.exterior.length

        uncov_norm = min(layout.uncovered_area / target_area, 1.0) if target_area else 0.0

        total_complexity = sum(
            cl.complexity
            for pp in layout.placed_pieces
            for cl in pp.cut_lines
        )
        max_complexity = target_perim * _COMPLEXITY_PERIM_RATIO
        complexity_norm = min(total_complexity / max_complexity, 1.0) if max_complexity else 0.0
        cuts_norm = min(layout.total_cuts / _MAX_CUTS, 1.0)
        cut_quality_norm = 0.5 * cuts_norm + 0.5 * complexity_norm

        if layout.remaining_target and not layout.remaining_target.is_empty:
            n_holes = len(_collect_polygons(layout.remaining_target))
            hole_norm = min(n_holes / _MAX_HOLE_COMPONENTS, 1.0)
        else:
            n_holes = 0
            hole_norm = 0.0
        fragmentation_norm = uncov_norm * (0.4 + 0.6 * hole_norm)

        total_piece_area = sum(pp.piece.area for pp in layout.placed_pieces)
        if total_piece_area > 0:
            material_waste_norm = min(layout.total_waste_area / total_piece_area, 1.0)
        elif target_area > 0:
            material_waste_norm = min(layout.total_waste_area / target_area, 1.0)
        else:
            material_waste_norm = 0.0

        coverage_penalty = uncov_norm ** 1.2

        return {
            "scale_factor": scale_factor,
            # Raw counts
            "num_cuts": layout.total_cuts,
            "total_cut_length": layout.total_cut_length,
            "total_cut_complexity": total_complexity,
            "uncovered_area": layout.uncovered_area,
            "uncovered_holes": n_holes,
            "waste_area": layout.total_waste_area,
            "total_piece_area": total_piece_area,
            "pieces_used": layout.num_pieces_used,
            "covered_area": layout.total_covered_area,
            # Percentages
            "coverage_pct": 100 * layout.coverage_ratio,
            "uncovered_pct": 100 * layout.uncovered_area / target_area if target_area else 0,
            "material_waste_pct": 100 * layout.total_waste_area / total_piece_area if total_piece_area else 0,
            # Normalised penalties
            "cuts_norm": cuts_norm,
            "complexity_norm": complexity_norm,
            "cut_quality_norm": cut_quality_norm,
            "hole_count": n_holes,
            "fragmentation_norm": fragmentation_norm,
            "uncov_norm": uncov_norm,
            "coverage_penalty": coverage_penalty,
            "material_waste_norm": material_waste_norm,
            # Final score
            "score": self.score(layout),
            "weights": {"w1": self.w1, "w2": self.w2, "w3": self.w3, "w4": self.w4},
        }


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

class BestLayoutSelector:
    """
    Accumulates FitnessResult objects from all scale iterations and exposes
    the winning configuration plus a human-readable report.
    """

    def __init__(self):
        self.results: List[FitnessResult] = []

    def add(self, result: FitnessResult) -> None:
        self.results.append(result)

    def add_all(self, results: List[FitnessResult]) -> None:
        self.results.extend(results)

    def best(self) -> Optional[FitnessResult]:
        return max(self.results, key=lambda r: r.score) if self.results else None

    def top_k(self, k: int = 3) -> List[FitnessResult]:
        return sorted(self.results, key=lambda r: r.score, reverse=True)[:k]

    def score_curve(self) -> List[Tuple[float, float]]:
        return sorted((r.scale_factor, r.score) for r in self.results)

    # ------------------------------------------------------------------
    def report(self, evaluator: Optional[FitnessEvaluator] = None) -> str:
        lines = [
            "=" * 70,
            "  MULTI-SCALE OPTIMISATION REPORT",
            "=" * 70,
            f"  {'Scale':>6}  {'Score':>8}  {'Cuts':>5}  "
            f"{'Coverage':>9}  {'Holes':>5}  {'WastePc':>7}  {'Pieces':>6}",
            "  " + "-" * 64,
        ]

        for r in sorted(self.results, key=lambda x: x.scale_factor):
            ly = r.layout
            ta = ly.target_polygon.area if ly.target_polygon else 1.0
            tp = sum(pp.piece.area for pp in ly.placed_pieces)
            cov_pct   = 100 * ly.coverage_ratio
            waste_pct = 100 * ly.total_waste_area / tp if tp else 0
            remaining = ly.remaining_target
            n_holes   = len(_collect_polygons(remaining)) if remaining and not remaining.is_empty else 0
            lines.append(
                f"  {r.scale_factor:>5.0%}  {r.score:>8.4f}  "
                f"{ly.total_cuts:>5d}  {cov_pct:>8.1f}%  "
                f"{n_holes:>5d}  {waste_pct:>6.1f}%  {ly.num_pieces_used:>6d}"
            )

        best = self.best()
        if best:
            ly = best.layout
            tp = sum(pp.piece.area for pp in ly.placed_pieces)
            ta = ly.target_polygon.area if ly.target_polygon else 1.0
            remaining = ly.remaining_target
            n_holes   = len(_collect_polygons(remaining)) if remaining and not remaining.is_empty else 0
            lines += [
                "=" * 70,
                f"  WINNER -> scale={best.scale_factor:.0%}  score={best.score:.4f}",
                f"  Pieces used        : {ly.num_pieces_used}",
                f"  Total cuts         : {ly.total_cuts}",
                f"  Cut length         : {ly.total_cut_length:.2f}",
                f"  Uncovered          : {ly.uncovered_area:.2f}  "
                f"({100*ly.uncovered_area/ta:.1f}%)",
                f"  Uncovered holes    : {n_holes}",
                f"  Waste (piece %)    : {100*ly.total_waste_area/tp:.1f}%" if tp else "  Waste: n/a",
                f"  Waste area         : {ly.total_waste_area:.2f}",
            ]

            lines.append("")
            lines.append("  Piece placements:")
            for i, pp in enumerate(ly.placed_pieces, 1):
                piece_waste_pct = 100 * pp.waste_area / pp.piece.area if pp.piece.area else 0
                lines.append(
                    f"    [{i:2d}] {pp.piece.piece_id:<22} "
                    f"rot={pp.rotation_degrees:>4.0f}°  "
                    f"pos=({pp.position[0]:>6.1f},{pp.position[1]:>6.1f})  "
                    f"covered={pp.covered_area:>7.1f}  "
                    f"waste={pp.waste_area:>7.1f} ({piece_waste_pct:.0f}%)  "
                    f"cuts={pp.num_cuts}"
                )

        lines.append("=" * 70)
        return "\n".join(lines)
