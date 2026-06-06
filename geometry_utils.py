"""
geometry_utils.py — Robust geometric primitives built on Shapely + OpenCV + SciPy.

All boolean operations defensively handle:
  - Invalid/degenerate polygons (make_valid)
  - MultiPolygon and GeometryCollection returns (extract largest)
  - Near-zero areas after floating-point arithmetic
  - Self-intersecting boundaries
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from shapely import affinity
from shapely.geometry import (
    GeometryCollection, LineString, MultiLineString,
    MultiPolygon, Point, Polygon,
)
from shapely.ops import unary_union
from shapely.validation import make_valid

from models import CutLine, EnvironmentBounds

logger = logging.getLogger(__name__)

# Floating-point tolerance for buffer-based cleanup
EPSILON = 1e-7


# ---------------------------------------------------------------------------
# Validity helpers
# ---------------------------------------------------------------------------

def _as_valid(geom):
    """Return a valid version of any Shapely geometry, or None if empty."""
    if geom is None or geom.is_empty:
        return geom
    if not geom.is_valid:
        geom = make_valid(geom)
    return geom


def _largest_polygon(geom) -> Optional[Polygon]:
    """
    Given any geometry (possibly Multi or Collection), return the single
    largest-area Polygon component, or an empty Polygon if none found.
    """
    if geom is None or geom.is_empty:
        return Polygon()

    if isinstance(geom, Polygon):
        return geom if geom.is_valid else _as_valid(geom)

    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        polys = [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
        if not polys:
            return Polygon()
        return max(polys, key=lambda p: p.area)

    return Polygon()


def _collect_polygons(geom) -> List[Polygon]:
    """Return all Polygon components from any geometry type."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom] if not geom.is_empty else []
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        return [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
    return []


# ---------------------------------------------------------------------------
# Boolean operations with full edge-case handling
# ---------------------------------------------------------------------------

def safe_intersection(poly_a: Polygon, poly_b: Polygon) -> Polygon:
    """
    Intersection of poly_a and poly_b.
    Returns the largest polygon component, or empty Polygon on failure.
    """
    try:
        a = _as_valid(poly_a)
        b = _as_valid(poly_b)
        if a is None or b is None or a.is_empty or b.is_empty:
            return Polygon()

        # Tiny inward buffer on both operands resolves boundary-coincident edge cases
        result = a.buffer(EPSILON).intersection(b.buffer(EPSILON)).buffer(-EPSILON)
        return _largest_polygon(_as_valid(result))

    except Exception as exc:
        logger.debug("safe_intersection failed: %s", exc)
        return Polygon()


def safe_difference(poly_a: Polygon, poly_b) -> Union[Polygon, MultiPolygon]:
    """
    poly_a minus poly_b.
    Returns poly_a unchanged on failure.  Preserves MultiPolygon when the
    difference produces disconnected regions.
    """
    try:
        a = _as_valid(poly_a)
        b = _as_valid(poly_b)
        if a is None or a.is_empty:
            return Polygon()
        if b is None or b.is_empty:
            return a

        result = _as_valid(a.difference(b))
        if result is None or result.is_empty:
            return Polygon()

        # Keep MultiPolygon so callers can iterate islands
        if isinstance(result, (Polygon, MultiPolygon)):
            return result

        # GeometryCollection — extract polygon parts
        polys = _collect_polygons(result)
        if not polys:
            return Polygon()
        return unary_union(polys) if len(polys) > 1 else polys[0]

    except Exception as exc:
        logger.debug("safe_difference failed: %s", exc)
        return poly_a


def safe_union(geometries: List) -> Union[Polygon, MultiPolygon]:
    """Union of a list of geometries; returns empty Polygon on failure."""
    try:
        valid = [_as_valid(g) for g in geometries if g and not g.is_empty]
        if not valid:
            return Polygon()
        return _as_valid(unary_union(valid))
    except Exception as exc:
        logger.debug("safe_union failed: %s", exc)
        return Polygon()


# ---------------------------------------------------------------------------
# Polygon construction
# ---------------------------------------------------------------------------

def polygon_from_contour(contour: np.ndarray) -> Optional[Polygon]:
    """Convert an OpenCV contour (N,1,2) array to a Shapely Polygon."""
    if contour is None or len(contour) < 3:
        return None
    pts = [(float(p[0][0]), float(p[0][1])) for p in contour]
    try:
        poly = Polygon(pts)
        return _as_valid(poly)
    except Exception as exc:
        logger.warning("polygon_from_contour: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Polygon transforms
# ---------------------------------------------------------------------------

def normalize_polygon(polygon: Polygon, bounds: EnvironmentBounds, margin: float = 0.95) -> Polygon:
    """
    Scale + translate a polygon so it fits inside the environment bounds,
    centred, with a fractional margin (default 5% inset from each edge).
    """
    minx, miny, maxx, maxy = polygon.bounds
    pw, ph = maxx - minx, maxy - miny
    if pw == 0 or ph == 0:
        return polygon

    scale = min(bounds.width / pw, bounds.height / ph) * margin

    coords = np.array(polygon.exterior.coords, dtype=float)
    coords -= [minx, miny]
    coords *= scale

    cx = (bounds.width - pw * scale) / 2
    cy = (bounds.height - ph * scale) / 2
    coords += [cx, cy]

    return _as_valid(Polygon(coords))


def scale_polygon(polygon: Polygon, factor: float, origin: str = "centroid") -> Polygon:
    return _as_valid(affinity.scale(polygon, xfact=factor, yfact=factor, origin=origin))


def rotate_polygon(polygon: Polygon, angle_deg: float, origin: str = "centroid") -> Polygon:
    return _as_valid(affinity.rotate(polygon, angle_deg, origin=origin))


def translate_polygon(polygon: Polygon, dx: float, dy: float) -> Polygon:
    return _as_valid(affinity.translate(polygon, xoff=dx, yoff=dy))


# ---------------------------------------------------------------------------
# Rasterization & Distance Transform
# ---------------------------------------------------------------------------

def rasterize_polygon(polygon, resolution: int = 512) -> np.ndarray:
    """
    Render a Shapely Polygon (or MultiPolygon) to a binary uint8 image.
    Interior pixels = 255, exterior = 0.
    """
    # Resolve to a single Polygon if needed
    if isinstance(polygon, MultiPolygon):
        polygon = max(polygon.geoms, key=lambda g: g.area)
    if not isinstance(polygon, Polygon) or polygon.is_empty:
        return np.zeros((resolution, resolution), dtype=np.uint8)

    minx, miny, maxx, maxy = polygon.bounds
    pw, ph = maxx - minx, maxy - miny
    if pw == 0 or ph == 0:
        return np.zeros((resolution, resolution), dtype=np.uint8)

    sx = resolution / pw
    sy = resolution / ph

    coords = np.array(polygon.exterior.coords, dtype=float)
    coords[:, 0] = (coords[:, 0] - minx) * sx
    coords[:, 1] = (coords[:, 1] - miny) * sy
    pts = coords.astype(np.int32)

    img = np.zeros((resolution, resolution), dtype=np.uint8)
    cv2.fillPoly(img, [pts], 255)
    return img


def compute_distance_transform(
    polygon,
    resolution: int = 256,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Euclidean distance transform of the polygon interior.

    Returns:
        dist_map - (H, W) float array; pixel value = distance to nearest boundary.
        meta     - (minx, miny, scale_x, scale_y) for pixel <-> world conversion.
    """
    if isinstance(polygon, MultiPolygon):
        polygon = max(polygon.geoms, key=lambda g: g.area)
    minx, miny, maxx, maxy = polygon.bounds
    pw, ph = maxx - minx, maxy - miny
    if pw == 0 or ph == 0:
        return np.zeros((resolution, resolution)), (minx, miny, 1.0, 1.0)

    sx, sy = resolution / pw, resolution / ph
    img = rasterize_polygon(polygon, resolution)
    dist_map = distance_transform_edt((img > 0).astype(np.uint8))
    return dist_map, (minx, miny, sx, sy)


def find_structural_zones(
    polygon: Polygon,
    n_zones: int = 8,
    resolution: int = 256,
) -> List[Tuple[float, float, float]]:
    """
    Identify the thickest interior zones of the polygon via its distance transform.
    These are the "medial" hotspots where the largest pieces should anchor.

    Returns list of (world_x, world_y, thickness_px) sorted thickest-first.
    The thickness value is in world units (not pixels).
    """
    dist_map, (minx, miny, sx, sy) = compute_distance_transform(polygon, resolution)

    zones: List[Tuple[float, float, float]] = []
    suppressed = np.zeros_like(dist_map, dtype=bool)

    for _ in range(n_zones):
        scratch = dist_map.copy()
        scratch[suppressed] = 0.0
        if scratch.max() < 1.0:
            break

        py, px = np.unravel_index(np.argmax(scratch), scratch.shape)
        thickness_px = float(dist_map[py, px])

        # Convert pixel coords back to world coords
        wx = px / sx + minx
        wy = py / sy + miny
        thickness_world = thickness_px / min(sx, sy)

        zones.append((wx, wy, thickness_world))

        # Suppress a disc around this zone so next iteration finds a new region
        radius = int(thickness_px * 1.5)
        y0, y1 = max(0, py - radius), min(resolution, py + radius)
        x0, x1 = max(0, px - radius), min(resolution, px + radius)
        suppressed[y0:y1, x0:x1] = True

    return sorted(zones, key=lambda z: z[2], reverse=True)


# ---------------------------------------------------------------------------
# Cut-line extraction
# ---------------------------------------------------------------------------

def extract_cut_lines(
    placed_polygon: Polygon,
    target_polygon: Polygon,
) -> List[CutLine]:
    """
    Determine where a physical cut must be made on `placed_polygon` to make
    it conform to `target_polygon`.

    A cut line is defined as any edge of the intersection polygon (covered area)
    that does NOT lie on the original piece boundary — i.e. it is a new edge
    introduced by the boolean clip.

    Strategy:
        cut_geometry = intersection.boundary  MINUS  placed_polygon.boundary
    """
    cut_lines: List[CutLine] = []
    try:
        placed = _as_valid(placed_polygon)
        target = _as_valid(target_polygon)
        if placed is None or target is None:
            return cut_lines

        intersection = safe_intersection(placed, target)
        if intersection.is_empty:
            return cut_lines

        # The new boundary segments introduced by the clip
        piece_boundary = placed.boundary
        inter_boundary = intersection.boundary

        # Grow piece boundary slightly to absorb floating-point coincidences
        cut_geom = inter_boundary.difference(piece_boundary.buffer(EPSILON * 100))

        if cut_geom is None or cut_geom.is_empty:
            return cut_lines

        # Decompose into individual LineString segments
        def _iter_lines(g):
            if isinstance(g, LineString):
                yield g
            elif isinstance(g, (MultiLineString, GeometryCollection)):
                for sub in g.geoms:
                    yield from _iter_lines(sub)

        for line in _iter_lines(cut_geom):
            if isinstance(line, LineString) and line.length > EPSILON * 1000:
                cut_lines.append(CutLine(geometry=line))

    except Exception as exc:
        logger.debug("extract_cut_lines failed: %s", exc)

    return cut_lines


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def polygon_area_safe(geom) -> float:
    """Return area of any geometry, safely handling None and Multis."""
    if geom is None or geom.is_empty:
        return 0.0
    return geom.area
