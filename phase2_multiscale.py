"""
phase2_multiscale.py — Phase 2: Multi-Scale Optimisation Loop

Rationale for sweeping scale:
  At 100% scale the target may be larger than any individual piece, forcing
  many cuts on every piece.  At 60% scale a single large rectangle might cover
  the torso completely (zero cuts).  The sweet spot — where piece geometry
  aligns with target zones — is different for every inventory and every prompt.
  Sweeping from scale_min to scale_max in small increments and scoring each
  layout with the Phase-4 fitness function finds that global optimum cheaply,
  since each layout evaluation is O(pieces x zones x rotations).
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon

from geometry_utils import _as_valid
from models import EnvironmentBounds, FitnessResult, InventoryPiece
from phase3_layout import GreedyPlacementEngine
from phase4_fitness import FitnessEvaluator

logger = logging.getLogger(__name__)


class MultiScaleOptimizer:
    """
    Sweeps the layout engine across a range of scale factors and collects
    a FitnessResult for each step.

    The base polygon passed to `run()` is assumed to already fill the
    environment bounds at scale=1.0.  The optimizer rescales it at each step
    and re-centres it inside the bounds.
    """

    def __init__(
        self,
        inventory: List[InventoryPiece],
        bounds: EnvironmentBounds,
        scale_min: float = 0.50,
        scale_max: float = 1.00,
        scale_step: float = 0.05,
        fitness_weights: Tuple[float, float, float, float] = (0.25, 0.15, 0.40, 0.20),
        layout_kwargs: dict | None = None,
    ):
        """
        Parameters
        ----------
        inventory       : Full piece inventory (passed to GreedyPlacementEngine).
        bounds          : Environment bounding box.
        scale_min/max   : Inclusive scale range (fractions, e.g. 0.5 -> 1.0).
        scale_step      : Increment between tested scales.
        fitness_weights : (w1, w2, w3, w4) forwarded to FitnessEvaluator.
        layout_kwargs   : Extra keyword arguments for GreedyPlacementEngine.
        """
        self.inventory = inventory
        self.bounds = bounds
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.scale_step = scale_step
        self.evaluator = FitnessEvaluator(weights=fitness_weights)
        self.fitness_weights = fitness_weights
        self.layout_kwargs = layout_kwargs or {}
        self.all_results: List[FitnessResult] = []

    # ------------------------------------------------------------------
    def run(self, base_polygon: Polygon) -> FitnessResult:
        """
        Run the full multi-scale optimisation and return the best result.

        Parameters
        ----------
        base_polygon : Target polygon normalised to fill bounds at scale=1.0.
        """
        # Build the scale grid (include scale_max even with float arithmetic)
        scales = np.arange(
            self.scale_min,
            self.scale_max + self.scale_step * 0.5,
            self.scale_step,
        )
        scales = np.clip(scales, self.scale_min, self.scale_max)

        logger.info(
            "Multi-scale optimisation: %d steps [%.0f%% -> %.0f%%]",
            len(scales), self.scale_min * 100, self.scale_max * 100,
        )

        self.all_results = []
        for scale in scales:
            result = self._evaluate_scale(base_polygon, float(scale))
            self.all_results.append(result)
            logger.info(
                "  scale=%4.0f%%  score=%7.4f  cuts=%2d  "
                "coverage=%5.1f%%  waste=%5.1f%%",
                scale * 100,
                result.score,
                result.layout.total_cuts,
                result.layout.coverage_ratio * 100,
                (100 * result.layout.total_waste_area / result.layout.target_polygon.area)
                if result.layout.target_polygon and result.layout.target_polygon.area
                else 0.0,
            )

        best = max(self.all_results, key=lambda r: r.score)
        logger.info(
            "Best: scale=%.0f%%  score=%.4f",
            best.scale_factor * 100, best.score,
        )
        return best

    # ------------------------------------------------------------------
    def _evaluate_scale(self, base_polygon: Polygon, scale: float) -> FitnessResult:
        target = self._scale_and_centre(base_polygon, scale)

        engine = GreedyPlacementEngine(
            inventory=self.inventory,
            target_polygon=target,
            bounds=self.bounds,
            **self.layout_kwargs,
        )
        layout = engine.run()
        s = self.evaluator.score(layout, scale)

        return FitnessResult(
            scale_factor=scale,
            score=s,
            layout=layout,
            weights=self.fitness_weights,
        )

    def _scale_and_centre(self, polygon: Polygon, scale: float) -> Polygon:
        """
        Scale `polygon` so it occupies `scale` fraction of the environment,
        preserving aspect ratio, then translate to centre within the bounds.
        """
        minx, miny, maxx, maxy = polygon.bounds
        pw, ph = maxx - minx, maxy - miny
        if pw == 0 or ph == 0:
            return polygon

        target_w = self.bounds.width * scale
        target_h = self.bounds.height * scale
        factor = min(target_w / pw, target_h / ph)

        scaled = affinity.scale(polygon, xfact=factor, yfact=factor, origin="centroid")

        s_minx, s_miny, s_maxx, s_maxy = scaled.bounds
        tx = (self.bounds.width - (s_maxx - s_minx)) / 2 - s_minx
        ty = (self.bounds.height - (s_maxy - s_miny)) / 2 - s_miny

        return _as_valid(affinity.translate(scaled, xoff=tx, yoff=ty))
