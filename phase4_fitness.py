"""
phase4_fitness.py — Phase 4: Fitness Scoring & Best Layout Selection

Fitness function (higher = better):

    Score = scale_factor
          - w1 * N_cuts_norm
          - w2 * cut_length_norm
          - w3 * uncovered_area_norm
          - w4 * waste_area_norm

All penalty terms are normalised to [0, 1] so the four weights are
directly comparable regardless of units or polygon size.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from models import FitnessResult, Layout

logger = logging.getLogger(__name__)

# Upper bounds used for normalisation.  Values above these are clamped to 1.
_MAX_CUTS = 30                # pieces rarely need more than this many cuts
_CUT_LENGTH_PERIMETER_RATIO = 8.0   # max expected cut_length / target_perimeter


class FitnessEvaluator:
    """
    Computes a scalar fitness score for a Layout at a given scale factor.

    Formula
    -------
        Score = scale_factor
              - w1 * cuts_norm           (cut complexity)
              - w2 * cut_length_norm     (cut effort)
              - w3 * uncovered_norm      (shape fidelity -- PRIMARY objective)
              - w4 * waste_norm          (material efficiency)

    All penalty terms are normalised to [0, 1] so weights are directly
    comparable regardless of polygon size or units.

    Coverage (w3) is the dominant term.  A large w3 (e.g. 0.80) means a
    15-point coverage gap outweighs a 10-point scale-factor gap, allowing
    a sub-100% scale step to win when inventory pieces fit it better.

    Parameters
    ----------
    weights : (w1, w2, w3, w4)
        Default (0.08, 0.04, 0.80, 0.08) prioritises coverage above all else.
    """

    def __init__(
        self,
        weights: Tuple[float, float, float, float] = (0.01, 0.01, 0.96, 0.02),
    ):
        self.w1, self.w2, self.w3, self.w4 = weights

    # ------------------------------------------------------------------
    def score(self, layout: Layout, scale_factor: float) -> float:
        """
        Compute and return the scalar fitness score.

        The four penalty terms are each individually normalised to [0, 1]:
          - cuts_norm     = min(n_cuts  / MAX_CUTS, 1)
          - cut_len_norm  = min(cut_len / (perimeter x ratio), 1)
          - uncov_norm    = uncovered_area / target_area
          - waste_norm    = waste_area    / target_area
        """
        if layout.target_polygon is None or layout.target_polygon.is_empty:
            return 0.0

        target_area = layout.target_polygon.area
        if target_area == 0:
            return 0.0
        target_perim = layout.target_polygon.exterior.length

        # ── Penalty 1: number of cuts ─────────────────────────────────
        cuts_norm = min(layout.total_cuts / _MAX_CUTS, 1.0)

        # ── Penalty 2: total cut length ───────────────────────────────
        max_cut_len = target_perim * _CUT_LENGTH_PERIMETER_RATIO
        cut_len_norm = (
            min(layout.total_cut_length / max_cut_len, 1.0)
            if max_cut_len > 0
            else 0.0
        )

        # ── Penalty 3: uncovered area ─────────────────────────────────
        uncov_norm = min(layout.uncovered_area / target_area, 1.0)

        # ── Penalty 4: waste area ─────────────────────────────────────
        waste_norm = min(layout.total_waste_area / target_area, 1.0)

        fitness = (
            scale_factor
            - self.w1 * cuts_norm
            - self.w2 * cut_len_norm
            - self.w3 * uncov_norm
            - self.w4 * waste_norm
        )

        logger.debug(
            "score | scale=%.2f | cuts=%.3fx%.2f | cut_len=%.3fx%.2f | "
            "uncov=%.3fx%.2f | waste=%.3fx%.2f -> %.4f",
            scale_factor,
            cuts_norm, self.w1,
            cut_len_norm, self.w2,
            uncov_norm, self.w3,
            waste_norm, self.w4,
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
        max_cut_len = target_perim * _CUT_LENGTH_PERIMETER_RATIO

        return {
            "scale_factor": scale_factor,
            # Raw counts
            "num_cuts": layout.total_cuts,
            "total_cut_length": layout.total_cut_length,
            "uncovered_area": layout.uncovered_area,
            "waste_area": layout.total_waste_area,
            "pieces_used": layout.num_pieces_used,
            "covered_area": layout.total_covered_area,
            # Percentages
            "coverage_pct": 100 * layout.coverage_ratio,
            "uncovered_pct": 100 * layout.uncovered_area / target_area if target_area else 0,
            "waste_pct": 100 * layout.total_waste_area / target_area if target_area else 0,
            # Normalised penalties
            "cuts_norm": min(layout.total_cuts / _MAX_CUTS, 1.0),
            "cut_len_norm": min(layout.total_cut_length / max_cut_len, 1.0) if max_cut_len else 0.0,
            "uncov_norm": min(layout.uncovered_area / target_area, 1.0) if target_area else 0.0,
            "waste_norm": min(layout.total_waste_area / target_area, 1.0) if target_area else 0.0,
            # Final score
            "score": self.score(layout, scale_factor),
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
        """Return the highest-scoring FitnessResult, or None if empty."""
        return max(self.results, key=lambda r: r.score) if self.results else None

    def top_k(self, k: int = 3) -> List[FitnessResult]:
        return sorted(self.results, key=lambda r: r.score, reverse=True)[:k]

    def score_curve(self) -> List[Tuple[float, float]]:
        """(scale, score) pairs sorted ascending by scale — useful for plotting."""
        return sorted((r.scale_factor, r.score) for r in self.results)

    # ------------------------------------------------------------------
    def report(self, evaluator: Optional[FitnessEvaluator] = None) -> str:
        """Multi-line human-readable summary of all scale iterations."""
        lines = [
            "=" * 65,
            "  MULTI-SCALE OPTIMISATION REPORT",
            "=" * 65,
            f"  {'Scale':>6}  {'Score':>8}  {'Cuts':>5}  "
            f"{'Coverage':>9}  {'Waste%':>7}  {'Pieces':>6}",
            "  " + "-" * 60,
        ]

        for r in sorted(self.results, key=lambda x: x.scale_factor):
            ly = r.layout
            ta = ly.target_polygon.area if ly.target_polygon else 1.0
            cov_pct = 100 * ly.coverage_ratio
            waste_pct = 100 * ly.total_waste_area / ta if ta else 0
            lines.append(
                f"  {r.scale_factor:>5.0%}  {r.score:>8.4f}  "
                f"{ly.total_cuts:>5d}  {cov_pct:>8.1f}%  "
                f"{waste_pct:>6.1f}%  {ly.num_pieces_used:>6d}"
            )

        best = self.best()
        if best:
            lines += [
                "=" * 65,
                f"  WINNER -> scale={best.scale_factor:.0%}  score={best.score:.4f}",
                f"  Pieces used   : {best.layout.num_pieces_used}",
                f"  Total cuts    : {best.layout.total_cuts}",
                f"  Cut length    : {best.layout.total_cut_length:.2f}",
                f"  Uncovered     : {best.layout.uncovered_area:.2f}  "
                f"({100*best.layout.uncovered_area / (best.layout.target_polygon.area or 1):.1f}%)",
                f"  Waste area    : {best.layout.total_waste_area:.2f}",
            ]

            # Per-piece detail
            lines.append("")
            lines.append("  Piece placements:")
            for i, pp in enumerate(best.layout.placed_pieces, 1):
                lines.append(
                    f"    [{i:2d}] {pp.piece.piece_id:<20} "
                    f"rot={pp.rotation_degrees:>4.0f}°  "
                    f"pos=({pp.position[0]:>6.1f},{pp.position[1]:>6.1f})  "
                    f"covered={pp.covered_area:>7.1f}  "
                    f"waste={pp.waste_area:>7.1f}  "
                    f"cuts={pp.num_cuts}"
                )

        lines.append("=" * 65)
        return "\n".join(lines)
