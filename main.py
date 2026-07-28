"""
main.py — Generative Design 2D Pipeline Entry Point

Orchestrates all four phases:
  Phase 1 — Silhouette generation + contour extraction
  Phase 2 — Multi-scale sweep
  Phase 3 — Greedy placement (called inside Phase 2)
  Phase 4 — Fitness scoring + best-layout selection (called inside Phase 2)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon
from shapely.validation import make_valid

from geometry_utils import normalize_polygon
from models import EnvironmentBounds, FitnessResult, InventoryPiece
from phase1_vision import ContourExtractor, ImageSilhouetteExtractor, SDXLTurboGenerator, SilhouetteGenerator
from phase2_multiscale import MultiScaleOptimizer
from phase4_fitness import BestLayoutSelector, FitnessEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inventory builder
# ---------------------------------------------------------------------------

def build_inventory() -> List[InventoryPiece]:
    """
    Inventory tuned for the multi-scale sweep (50 → 100 %, step 10 %).
    Fitness = -w1*cuts -w2*cut_len -w3*uncovered -w4*waste  (scale not in formula).

    Design logic
    ------------
    For 90% to be the winner, the coverage at 90% must be substantially
    higher than at 100%, so the w3*uncov_norm penalty makes 90% win:
      P(100%) contains  0.91 * 0.30 = 0.273  just from uncoverage
      P(90%)  contains  0.91 * 0.05 = 0.046  when coverage is 95%

    To force that coverage gap:
      * Total material = ~112 000 sq units
      * House area at 90% scale  ~= 107 000  -> material slightly exceeds target
        -> near-100% coverage achievable at 90%
      * House area at 100% scale ~= 160 000  -> material covers only ~70%
        -> large uncoverage penalty at 100%

    Piece sizing (house bbox ~= 450x450 at 90% scale, factor=0.947):
      body_slab    : 70% x 47% of 450  = 315 x 211   (covers wall section exactly)
      roof_tri_L/R : each half-triangle = 189 base x 171 height
      chimney      : 10% x 24% of 450  =  45 x 108
      two fillers  : 100x75 and 50x25  (cover eave residuals)
    """

    def rect(w, h):
        return Polygon([(0, 0), (w, 0), (w, h), (0, h)])

    # All primary pieces are scaled to 98% of the exact 90%-scale house zone
    # dimensions.  The 2% inset gives a margin so pieces land completely INSIDE
    # their zone -> zero waste, zero cuts at 90%.  At 100% the house is 11%
    # larger, so the same pieces cover only ~66% -> large uncoverage penalty.
    pieces = [
        # body_slab: 98% of (315 x 211)  ->  fits body interior with ~3 unit margin
        InventoryPiece("body_slab",     rect(309, 207),                    "plywood"),  # 63 963
        # roof halves: 98% of (189 x 171) legs
        InventoryPiece("roof_tri_L",    Polygon([(0,0),(185,0),(0,168)]),  "plywood"),  # 15 540
        InventoryPiece("roof_tri_R",    Polygon([(0,0),(185,0),(185,168)]),"plywood"),  # 15 540
        # chimney: 98% of (45 x 108)
        InventoryPiece("chimney_block", rect(44, 106),                     "plywood"),  #  4 664

        # Four 40x40 fill squares to close residual gaps at 90%
        InventoryPiece("fill_sq_1",     rect(40, 40),  "plywood"),  # 1 600
        InventoryPiece("fill_sq_2",     rect(40, 40),  "plywood"),  # 1 600
        InventoryPiece("fill_sq_3",     rect(40, 40),  "plywood"),  # 1 600
        InventoryPiece("fill_sq_4",     rect(40, 40),  "plywood"),  # 1 600
    ]
    # Grand total: ~106 107
    # house@90%  ~= 107 000  -> material ~= target  -> near-100% coverage, zero cuts
    # house@100% ~= 160 000  -> material covers only ~66%  -> large uncoverage penalty

    total = sum(p.area for p in pieces)
    logger.info(
        "Inventory: %d pieces, total area=%.0f  "
        "(house@90%% ~107 000 | house@100%% ~160 000)",
        len(pieces), total,
    )
    return pieces


# ---------------------------------------------------------------------------
# JSON inventory loader  (reads from inventory_pieces/ folder)
# ---------------------------------------------------------------------------

#: Folder where scrap JSON files are dropped.  Created automatically if absent.
INVENTORY_FOLDER = Path(__file__).parent / "inventory_pieces"

#: Physical width of the target object in cm.
#: Used to scale piece cm measurements into pipeline world units.
#: Adjust this to match the real-world size of your target shape.
TARGET_PHYSICAL_WIDTH_CM: float = 20


def load_inventory_from_folder(
    folder: Path = INVENTORY_FOLDER,
    world_units_per_cm: float | None = None,
    bounds: EnvironmentBounds | None = None,
) -> List[InventoryPiece]:
    """
    Read every ``*.json`` file in *folder*, convert scrap detections to
    InventoryPiece objects, and return them as a flat list.

    Each JSON must follow the scrap-detector output format::

        {
          "cm_per_pixel": <float>,
          "scraps": [
            {
              "id": <int|str>,
              "contour_points": [[x, y], ...],   # pixel coords, y-down
              ...
            },
            ...
          ]
        }

    Parameters
    ----------
    folder            : Directory to scan for JSON files (created if missing).
    world_units_per_cm: Scale factor cm → pipeline world units.
                        Defaults to  BOUNDS.width / TARGET_PHYSICAL_WIDTH_CM.
    bounds            : EnvironmentBounds used only when world_units_per_cm is None.
    """
    folder.mkdir(parents=True, exist_ok=True)

    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        logger.warning(
            "inventory_pieces/ folder is empty — falling back to built-in inventory."
        )
        return []

    if world_units_per_cm is None:
        b = bounds or EnvironmentBounds(width=600, height=500)
        world_units_per_cm = b.width / TARGET_PHYSICAL_WIDTH_CM

    pieces: List[InventoryPiece] = []

    for json_path in json_files:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Skipping %s — could not parse JSON: %s", json_path.name, exc)
            continue

        cm_per_pixel = data.get("cm_per_pixel", 1.0)

        for scrap in data.get("scraps", []):
            scrap_id = scrap.get("id", "?")
            pts_px = scrap.get("contour_points", [])

            if len(pts_px) < 3:
                logger.warning("scrap %s in %s has < 3 points — skipped.",
                               scrap_id, json_path.name)
                continue

            try:
                # 1. Pixel → cm
                pts_cm = [(x * cm_per_pixel, y * cm_per_pixel) for x, y in pts_px]

                # 2. Flip Y  (image: y=0 top  →  math: y=0 bottom)
                max_y_cm = max(y for _, y in pts_cm)
                pts_flipped = [(x, max_y_cm - y) for x, y in pts_cm]

                # 3. Build and validate polygon
                poly = make_valid(Polygon(pts_flipped))
                if poly.is_empty or poly.area <= 0:
                    logger.warning("scrap %s in %s produced empty polygon — skipped.",
                                   scrap_id, json_path.name)
                    continue

                # 4. Scale cm → world units
                poly = affinity.scale(
                    poly,
                    xfact=world_units_per_cm,
                    yfact=world_units_per_cm,
                    origin=(0, 0),
                )

                piece_id = f"{json_path.stem}_scrap_{scrap_id}"
                pieces.append(InventoryPiece(
                    piece_id=piece_id,
                    polygon=poly,
                    material=scrap.get("material", "scrap"),
                ))

            except Exception as exc:
                logger.warning("scrap %s in %s could not be converted: %s",
                               scrap_id, json_path.name, exc)

    logger.info(
        "Loaded %d pieces from %d JSON file(s) in '%s'",
        len(pieces), len(json_files), folder.name,
    )
    return pieces


# ---------------------------------------------------------------------------
# Visualiser (matplotlib, optional)
# ---------------------------------------------------------------------------

def visualize(best: FitnessResult, out_path: str = "output_layout.png") -> None:
    """Render the winning layout to a PNG file."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path as MPath

        def _poly_patch(poly, **kw):
            """Convert a Shapely polygon to a matplotlib PathPatch."""
            coords = list(poly.exterior.coords)
            codes = [MPath.MOVETO] + [MPath.LINETO] * (len(coords) - 2) + [MPath.CLOSEPOLY]
            return PathPatch(MPath(coords, codes), **kw)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        layout = best.layout

        for ax, title_suffix in zip(axes, ["with cut-lines", "clean layout"]):
            ax.set_aspect("equal")
            ax.set_facecolor("#f5f5f0")

            # Target outline
            if layout.target_polygon and not layout.target_polygon.is_empty:
                ax.add_patch(_poly_patch(layout.target_polygon,
                                         facecolor="#ddeeff", edgecolor="#2266cc",
                                         linewidth=2, zorder=1))

            colors = cm.Set2(np.linspace(0, 1, max(len(layout.placed_pieces), 1)))

            for i, pp in enumerate(layout.placed_pieces):
                c = colors[i]

                # Covered portion
                if pp.intersection_with_target and not pp.intersection_with_target.is_empty:
                    ax.add_patch(_poly_patch(pp.intersection_with_target,
                                             facecolor=(*c[:3], 0.75),
                                             edgecolor="k", linewidth=0.6, zorder=2))

                # Waste (shown only in left panel)
                if title_suffix == "with cut-lines" and pp.waste_polygon \
                        and not pp.waste_polygon.is_empty:
                    ax.add_patch(_poly_patch(pp.waste_polygon,
                                             facecolor=(1, 0.3, 0.3, 0.25),
                                             edgecolor=(0.7, 0, 0), linewidth=0.5,
                                             linestyle="--", zorder=2))

                # Cut lines (left panel only)
                if title_suffix == "with cut-lines":
                    for cl in pp.cut_lines:
                        xs, ys = cl.geometry.xy
                        ax.plot(xs, ys, color="red", linewidth=1.8, zorder=5,
                                solid_capstyle="round")

                # Label
                if pp.intersection_with_target and not pp.intersection_with_target.is_empty:
                    cx2, cy2 = pp.intersection_with_target.centroid.coords[0]
                    ax.text(cx2, cy2, pp.piece.piece_id.replace("_", "\n"),
                            ha="center", va="center", fontsize=6, zorder=6,
                            color="#222222")

            # Uncovered residual
            if layout.remaining_target and not layout.remaining_target.is_empty:
                if hasattr(layout.remaining_target, "exterior"):
                    ax.add_patch(_poly_patch(layout.remaining_target,
                                             facecolor=(1, 0.65, 0, 0.35),
                                             edgecolor=(0.8, 0.4, 0), linewidth=1,
                                             hatch="////", zorder=3))

            ta = layout.target_polygon.area if layout.target_polygon else 1
            ax.set_title(
                f"Scale {best.scale_factor:.0%} | Score {best.score:.4f} | "
                f"Cuts {layout.total_cuts} | "
                f"Coverage {layout.coverage_ratio*100:.1f}% | "
                f"Waste {layout.total_waste_area/ta*100:.1f}%\n({title_suffix})",
                fontsize=9,
            )
            ax.autoscale_view()
            ax.margins(0.05)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Layout saved -> %s", out_path)

    except ImportError:
        logger.warning("matplotlib not available — skipping visualisation.")
    except Exception as exc:
        logger.warning("Visualisation failed: %s", exc)


def visualize_all(
    results: List[FitnessResult],
    best: FitnessResult,
    out_path: str = "all_scales_layout.png",
) -> None:
    """Render all scale layouts in a grid so every scale is visible at once."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path as MPath

        def _poly_patch(poly, **kw):
            coords = list(poly.exterior.coords)
            codes = [MPath.MOVETO] + [MPath.LINETO] * (len(coords) - 2) + [MPath.CLOSEPOLY]
            return PathPatch(MPath(coords, codes), **kw)

        n = len(results)
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
        axes_flat = np.array(axes).flatten()

        sorted_results = sorted(results, key=lambda r: r.scale_factor)

        for idx, result in enumerate(sorted_results):
            ax = axes_flat[idx]
            ax.set_aspect("equal")
            ax.set_facecolor("#f5f5f0")
            layout = result.layout
            is_best = abs(result.scale_factor - best.scale_factor) < 1e-6

            if layout.target_polygon and not layout.target_polygon.is_empty:
                edge_color = "#cc2222" if is_best else "#2266cc"
                lw = 2.5 if is_best else 1.5
                ax.add_patch(_poly_patch(layout.target_polygon,
                                         facecolor="#ddeeff", edgecolor=edge_color,
                                         linewidth=lw, zorder=1))

            colors = cm.Set2(np.linspace(0, 1, max(len(layout.placed_pieces), 1)))
            for i, pp in enumerate(layout.placed_pieces):
                if pp.intersection_with_target and not pp.intersection_with_target.is_empty:
                    ax.add_patch(_poly_patch(pp.intersection_with_target,
                                             facecolor=(*colors[i][:3], 0.75),
                                             edgecolor="k", linewidth=0.5, zorder=2))
                    cx, cy = pp.intersection_with_target.centroid.coords[0]
                    ax.text(cx, cy, pp.piece.piece_id.replace("_", "\n"),
                            ha="center", va="center", fontsize=5, zorder=6,
                            color="#222222")

            if layout.remaining_target and not layout.remaining_target.is_empty:
                if hasattr(layout.remaining_target, "exterior"):
                    ax.add_patch(_poly_patch(layout.remaining_target,
                                             facecolor=(1, 0.65, 0, 0.35),
                                             edgecolor=(0.8, 0.4, 0), linewidth=0.8,
                                             hatch="////", zorder=3))

            cov = layout.coverage_ratio * 100
            star = " ★ BEST" if is_best else ""
            ax.set_title(
                f"Scale {result.scale_factor:.0%}{star}\n"
                f"Score {result.score:.4f} | Coverage {cov:.1f}% | "
                f"Cuts {layout.total_cuts}",
                fontsize=8,
                color="#cc2222" if is_best else "black",
                fontweight="bold" if is_best else "normal",
            )
            ax.autoscale_view()
            ax.margins(0.05)

        # Hide unused subplots
        for idx in range(len(sorted_results), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.suptitle("All Scale Layouts", fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("All-scales layout saved -> %s", out_path)

    except ImportError:
        logger.warning("matplotlib not available — skipping visualisation.")
    except Exception as exc:
        logger.warning("visualize_all failed: %s", exc)


def save_per_piece_cuts(
    best: FitnessResult,
    out_dir: str = "cuts_output",
    world_units_per_cm: float | None = None,
) -> None:
    """
    For every placed piece in the winning layout, write two files into *out_dir*:

      {piece_id}.png  — diagram: full piece footprint, kept area, waste, cut lines
      {piece_id}.json — machine-readable cut-line coordinates and measurements
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    # Scale factor for displaying lengths in cm (optional, purely cosmetic)
    wu_per_cm = world_units_per_cm or (600 / TARGET_PHYSICAL_WIDTH_CM)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path as MPath

        def _poly_patch(poly, **kw):
            if poly is None or poly.is_empty:
                return None
            coords = list(poly.exterior.coords)
            codes = ([MPath.MOVETO]
                     + [MPath.LINETO] * (len(coords) - 2)
                     + [MPath.CLOSEPOLY])
            return PathPatch(MPath(coords, codes), **kw)

        layout = best.layout

        for pp in layout.placed_pieces:
            pid = pp.piece.piece_id
            safe_pid = pid.replace("/", "_").replace("\\", "_")

            # ── PNG ───────────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_aspect("equal")
            ax.set_facecolor("#f8f8f5")

            # Full piece outline (light fill so cuts stand out)
            patch = _poly_patch(pp.transformed_polygon,
                                 facecolor="#d0e8ff", edgecolor="#336699",
                                 linewidth=1.2, zorder=1, label="Full piece")
            if patch:
                ax.add_patch(patch)

            # Waste zone (part that gets cut away)
            if pp.waste_polygon and not pp.waste_polygon.is_empty:
                patch = _poly_patch(pp.waste_polygon,
                                     facecolor="#ffcccc", edgecolor="#cc3300",
                                     linewidth=0.8, linestyle="--", zorder=2,
                                     label="Waste (cut away)", alpha=0.6)
                if patch:
                    ax.add_patch(patch)

            # Kept area (intersection with target)
            if pp.intersection_with_target and not pp.intersection_with_target.is_empty:
                patch = _poly_patch(pp.intersection_with_target,
                                     facecolor="#66bb88", edgecolor="#225533",
                                     linewidth=1.0, zorder=3, alpha=0.75,
                                     label="Kept (inside target)")
                if patch:
                    ax.add_patch(patch)

            # Cut lines — drawn thick in red, annotated with length
            for idx, cl in enumerate(pp.cut_lines):
                xs, ys = cl.geometry.xy
                ax.plot(xs, ys, color="#dd0000", linewidth=2.5, zorder=5,
                        solid_capstyle="round",
                        label="Cut line" if idx == 0 else "_")
                # Length label at midpoint
                mx = (xs[0] + xs[-1]) / 2
                my = (ys[0] + ys[-1]) / 2
                length_cm = cl.length / wu_per_cm
                ax.text(mx, my, f"{length_cm:.1f} cm",
                        fontsize=7, color="#aa0000", zorder=6,
                        ha="center", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.7))

            covered_cm2 = pp.covered_area / (wu_per_cm ** 2)
            waste_cm2   = pp.waste_area   / (wu_per_cm ** 2)
            ax.set_title(
                f"{pid}\n"
                f"Rotation: {pp.rotation_degrees:.0f}°  |  "
                f"Cuts: {pp.num_cuts}  |  "
                f"Total cut length: {pp.total_cut_length / wu_per_cm:.1f} cm\n"
                f"Kept: {covered_cm2:.2f} cm²   Waste: {waste_cm2:.2f} cm²",
                fontsize=9,
            )

            handles, labels = ax.get_legend_handles_labels()
            # deduplicate legend
            seen = {}
            for h, l in zip(handles, labels):
                if l not in seen:
                    seen[l] = h
            ax.legend(seen.values(), seen.keys(), fontsize=7, loc="lower right")

            ax.autoscale_view()
            ax.margins(0.10)
            fig.tight_layout()
            png_path = os.path.join(out_dir, f"{safe_pid}.png")
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            # ── JSON ──────────────────────────────────────────────────
            cuts_data = []
            for idx, cl in enumerate(pp.cut_lines):
                coords = list(cl.geometry.coords)
                cuts_data.append({
                    "cut_index": idx,
                    "length_world": round(cl.length, 4),
                    "length_cm": round(cl.length / wu_per_cm, 4),
                    "is_straight": cl.is_straight,
                    "num_vertices": len(coords),
                    "coordinates_world": [[round(x, 3), round(y, 3)]
                                          for x, y in coords],
                })

            piece_data = {
                "piece_id": pid,
                "rotation_degrees": pp.rotation_degrees,
                "position_dx": round(pp.position[0], 3),
                "position_dy": round(pp.position[1], 3),
                "covered_area_world": round(pp.covered_area, 3),
                "covered_area_cm2":   round(covered_cm2, 4),
                "waste_area_world":   round(pp.waste_area, 3),
                "waste_area_cm2":     round(waste_cm2, 4),
                "num_cuts": pp.num_cuts,
                "total_cut_length_world": round(pp.total_cut_length, 3),
                "total_cut_length_cm":    round(pp.total_cut_length / wu_per_cm, 4),
                "cuts": cuts_data,
            }

            json_path = os.path.join(out_dir, f"{safe_pid}.json")
            Path(json_path).write_text(
                json.dumps(piece_data, indent=2), encoding="utf-8"
            )
            logger.info("  Saved cut files: %s  (cuts=%d)", safe_pid, pp.num_cuts)

        logger.info("Per-piece cut files written to '%s/'", out_dir)

    except ImportError:
        logger.warning("matplotlib not available — skipping per-piece cut diagrams.")
    except Exception as exc:
        logger.warning("save_per_piece_cuts failed: %s", exc)


def plot_score_curve(selector: BestLayoutSelector, out_path: str = "score_curve.png") -> None:
    """Line plot of score vs scale factor across all iterations."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        curve = selector.score_curve()
        xs = [c[0] * 100 for c in curve]
        ys = [c[1] for c in curve]

        best = selector.best()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(xs, ys, "o-", color="#2266cc", linewidth=2, markersize=6)
        if best:
            ax.axvline(best.scale_factor * 100, color="red", linestyle="--",
                       label=f"Best: {best.scale_factor:.0%}")
        ax.set_xlabel("Scale (%)")
        ax.set_ylabel("Fitness Score")
        ax.set_title("Fitness Score vs Scale Factor")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        logger.info("Score curve saved -> %s", out_path)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _prompt_interface() -> dict:
    """
    Interactive CLI that asks the user how to source the target silhouette.
    Returns a config dict consumed by main().
    """
    # Drain buffered Enter key from launching the script in Windows terminals
    try:
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()
    except ImportError:
        pass

    DEFAULT_IMAGE = r"C:\Users\dhara\OneDrive\Desktop\GENERATIVE_DESIGN_2D\test_house_image.avif"

    print("\n" + "=" * 60)
    print("  GENERATIVE DESIGN 2D")
    print("=" * 60)
    print("\nSelect input mode:")
    print("  [1] Existing image  — extract silhouette from a photo/file")
    print("  [2] SDXL-Turbo      — generate image from a text prompt")
    print("  [3] SDXL (full)     — generate silhouette via full SDXL")
    print()

    while True:
        choice = input("Choice [1/2/3, default=1]: ").strip() or "1"
        if choice in ("1", "2", "3"):
            break
        print("  Enter 1, 2, or 3.")

    if choice == "1":
        raw = input(f"Image path [default: {DEFAULT_IMAGE}]: ").strip()
        image_path = raw if raw else DEFAULT_IMAGE
        raw_prompt = input("Object prompt for detection [default: house]: ").strip()
        prompt = raw_prompt if raw_prompt else "house"
        return {"mode": "image", "image_path": image_path, "prompt": prompt}

    if choice == "2":
        prompt = ""
        while not prompt:
            prompt = input('Text prompt (e.g. "house", "car", "tree"): ').strip()
            if not prompt:
                print("  Prompt cannot be empty.")
        raw_steps = input("Inference steps [1-4, default=2]: ").strip()
        try:
            steps = max(1, min(4, int(raw_steps)))
        except ValueError:
            steps = 2
        return {"mode": "turbo", "prompt": prompt, "steps": steps}

    # choice == "3"
    prompt = ""
    while not prompt:
        prompt = input('Text prompt (e.g. "house", "bicycle"): ').strip()
        if not prompt:
            print("  Prompt cannot be empty.")
    return {"mode": "sdxl", "prompt": prompt}


def main() -> FitnessResult | None:
    cfg = _prompt_interface()
    mode   = cfg["mode"]
    PROMPT = cfg.get("prompt", "house")
    BOUNDS = EnvironmentBounds(width=600, height=500)

    logger.info("=" * 65)
    logger.info("  GENERATIVE DESIGN 2D PIPELINE  mode=%s  prompt='%s'",
                mode, PROMPT)
    logger.info("=" * 65)

    FITNESS_WEIGHTS = (
        0.05,   # w1 — cut quality (count + complexity)
        0.04,   # w2 — fragmentation of uncovered area
        0.85,   # w3 — coverage, non-linear (dominant)
        0.06,   # w4 — material efficiency (waste / piece area)
    )

    SCALE_STEP = 0.10   # steps: 50 → 60 → 70 → 80 → 90 → 100 %

    LAYOUT_KWARGS = {
        "n_structural_zones": 6,
        "rotation_candidates": [0, 90, 180, 270],
        "min_remaining_area": 50.0,
    }

    # ── Phase 1: Silhouette & Contour ────────────────────────────────
    logger.info("\n[Phase 1] Silhouette generation + contour extraction")

    if mode == "image":
        logger.info("  Mode: image  (source: %s)", cfg["image_path"])
        extractor_img = ImageSilhouetteExtractor(output_size=(512, 512))
        silhouette = extractor_img.extract_from_image(
            image_path=cfg["image_path"],
            prompt=PROMPT,
            output_path="silhouette.png",
        )
    elif mode == "turbo":
        logger.info("  Mode: SDXL-Turbo  (prompt: '%s', steps: %d)",
                    PROMPT, cfg["steps"])
        gen = SDXLTurboGenerator(output_size=(512, 512),
                                 num_inference_steps=cfg["steps"])
        silhouette = gen.generate(prompt=PROMPT, output_path="silhouette.png")
    else:
        logger.info("  Mode: SDXL full  (prompt: '%s')", PROMPT)
        gen = SilhouetteGenerator(output_size=(512, 512))
        silhouette = gen.generate(prompt=PROMPT, output_path="silhouette.png")

    logger.info("  Image shape: %s", silhouette.shape)

    extractor = ContourExtractor(epsilon_factor=0.008, min_area_ratio=0.015)
    target_poly, raw_contour = extractor.extract(silhouette)

    if target_poly is None:
        logger.error("Contour extraction failed — aborting.")
        return None

    logger.info(
        "  Target polygon: %d vertices, area=%.1f",
        len(target_poly.exterior.coords), target_poly.area,
    )

    # Save contour debug image
    extractor.visualize(silhouette, raw_contour, output_path="contour_debug.png")

    # Normalise to environment bounds at scale = 1.0
    target_norm = normalize_polygon(target_poly, BOUNDS)
    logger.info("  Normalised bounds: %s", [f"{v:.1f}" for v in target_norm.bounds])

    # ── Phase 2: Multi-Scale Sweep (contains Phase 3 + 4 calls) ─────
    logger.info("\n[Phase 2] Multi-scale optimisation sweep")

    # Load from inventory_pieces/ folder; fall back to built-in if folder is empty
    inventory = load_inventory_from_folder(bounds=BOUNDS)
    if not inventory:
        logger.info("  Using built-in inventory (no JSON files found).")
        inventory = build_inventory()
    logger.info("  Inventory: %d pieces", len(inventory))

    optimizer = MultiScaleOptimizer(
        inventory=inventory,
        bounds=BOUNDS,
        scale_min=0.50,
        scale_max=1.00,
        scale_step=SCALE_STEP,
        fitness_weights=FITNESS_WEIGHTS,
        layout_kwargs=LAYOUT_KWARGS,
    )

    best = optimizer.run(target_norm)

    # ── Phase 4: Results ─────────────────────────────────────────────
    logger.info("\n[Phase 4] Results")

    selector = BestLayoutSelector()
    selector.add_all(optimizer.all_results)

    print("\n" + selector.report())

    evaluator = FitnessEvaluator(weights=FITNESS_WEIGHTS)
    bd = evaluator.breakdown(best.layout, best.scale_factor)
    print("\nDETAILED BREAKDOWN (winning layout):")
    col_w = max(len(k) for k in bd) + 2
    for k, v in bd.items():
        if isinstance(v, float):
            print(f"  {k:<{col_w}}: {v:.4f}")
        else:
            print(f"  {k:<{col_w}}: {v}")

    # ── Outputs ──────────────────────────────────────────────────────
    visualize(best, out_path="output_layout.png")
    visualize_all(optimizer.all_results, best, out_path="all_scales_layout.png")
    plot_score_curve(selector, out_path="score_curve.png")
    save_per_piece_cuts(best, out_dir="cuts_output")

    logger.info("\nDone.  Output files: silhouette.png, contour_debug.png, "
                "output_layout.png, all_scales_layout.png, score_curve.png, "
                "cuts_output/{piece_id}.png + .json")
    return best


if __name__ == "__main__":
    main()
