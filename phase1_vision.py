"""
phase1_vision.py — Phase 1: Silhouette Generation & Contour Extraction

Two-step:
  1. SilhouetteGenerator  – text prompt -> binary (B/W) image
     Attempts SDXL via diffusers; falls back to a synthetic dog polygon.
  2. ContourExtractor     – binary image -> simplified Shapely Polygon
     Uses cv2.findContours + Douglas-Peucker (cv2.approxPolyDP).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
from shapely.geometry import Polygon

from geometry_utils import polygon_from_contour, _as_valid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Image generation
# ---------------------------------------------------------------------------

class SilhouetteGenerator:
    """
    Produces a binary silhouette image (uint8, HxW) for a text prompt.

    When use_diffusion=True, it attempts to load SDXL from HuggingFace and
    generate a pure black silhouette on white background via a carefully
    crafted prompt + negative prompt.  Falls back to a hand-crafted synthetic
    dog silhouette if the library is unavailable or any error occurs.
    """

    SILHOUETTE_POSITIVE = (
        "Pure solid black silhouette of {subject} on pure white background, "
        "flat design, no shading, no gradient, no texture, no outline only, "
        "completely filled solid black shape, vector art style, high contrast"
    )
    SILHOUETTE_NEGATIVE = (
        "gray, shadow, gradient, texture, multiple subjects, "
        "white fill, outline only, sketch, watercolor, realistic"
    )

    def __init__(
        self,
        use_diffusion: bool = True,
        output_size: Tuple[int, int] = (512, 512),
        num_inference_steps: int = 30,
        guidance_scale: float = 9.0,
    ):
        self.use_diffusion = use_diffusion
        self.output_size = output_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self._pipe = None   # lazy-loaded

    # ------------------------------------------------------------------
    def generate(self, prompt: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        Generate and return a binary silhouette image (0=foreground, 255=background).
        Optionally saves it to `output_path`.
        """
        if self.use_diffusion:
            self._ensure_pipeline()

        img = (
            self._from_diffusion(prompt)
            if self.use_diffusion
            else self._synthetic_dispatch(prompt)
        )

        if output_path:
            cv2.imwrite(output_path, img)
            logger.info("Silhouette saved -> %s", output_path)

        return img

    # ------------------------------------------------------------------
    def _ensure_pipeline(self):
        if self._pipe is not None:
            return
        try:
            import torch
            from diffusers import StableDiffusionXLPipeline

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            logger.info("Loading SDXL on %s (dtype=%s)…", device, dtype)

            self._pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=dtype,
                use_safetensors=True,
                variant="fp16" if device == "cuda" else None,
            ).to(device)

            logger.info("SDXL ready.")
        except Exception as exc:
            logger.warning("SDXL unavailable (%s). Using synthetic fallback.", exc)
            self.use_diffusion = False

    def _from_diffusion(self, prompt: str) -> np.ndarray:
        w, h = self.output_size
        pos = self.SILHOUETTE_POSITIVE.format(subject=prompt)
        neg = self.SILHOUETTE_NEGATIVE

        result = self._pipe(
            prompt=pos,
            negative_prompt=neg,
            width=w, height=h,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        )
        pil_img = result.images[0].convert("L")   # grayscale
        gray = np.array(pil_img, dtype=np.uint8)

        # Otsu threshold -> binary: dark ink = foreground (255), white bg = 0
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    # ------------------------------------------------------------------
    def _synthetic_dispatch(self, prompt: str) -> np.ndarray:
        """Route to the correct synthetic generator based on prompt keyword."""
        p = prompt.lower()
        if "house" in p or "home" in p or "building" in p:
            return self._synthetic_house()
        return self._synthetic_dog()

    # ------------------------------------------------------------------
    def _synthetic_house(self) -> np.ndarray:
        """
        Hand-crafted front-view house silhouette using OpenCV drawing primitives.
        Shape: rectangular body + triangular roof + chimney + door arch.
        All measurements are fractions of (w, h).
        """
        w, h = self.output_size
        img = np.full((h, w), 255, dtype=np.uint8)   # white background

        def ip(fx, fy):
            return (int(fx * w), int(fy * h))

        def ia(fx, fy):
            return (int(fx * w), int(fy * h))

        # ── Main body (tall rectangle) ────────────────────────────────
        body_tl = ip(0.15, 0.45)
        body_br = ip(0.85, 0.92)
        cv2.rectangle(img, body_tl, body_br, 0, -1)

        # ── Roof (isoceles triangle sitting on top of body) ───────────
        roof = np.array([
            ip(0.08, 0.46),    # left eave
            ip(0.92, 0.46),    # right eave
            ip(0.50, 0.08),    # apex
        ], dtype=np.int32)
        cv2.fillPoly(img, [roof], 0)

        # ── Chimney (rectangle rising from left side of roof) ─────────
        chimney = np.array([
            ip(0.28, 0.08),
            ip(0.38, 0.08),
            ip(0.38, 0.32),
            ip(0.28, 0.32),
        ], dtype=np.int32)
        cv2.fillPoly(img, [chimney], 0)

        # ── Door (rounded arch: rectangle + semicircle top) ──────────
        door_tl = ip(0.42, 0.68)
        door_br = ip(0.58, 0.92)
        cv2.rectangle(img, door_tl, door_br, 255, -1)   # cut door OUT (white)

        door_cx = int(0.50 * w)
        door_cy = int(0.68 * h)
        door_rx = int(0.08 * w)
        cv2.ellipse(img, (door_cx, door_cy), (door_rx, door_rx),
                    0, 180, 360, 255, -1)                # arch top (white)

        # ── Left window (square) ──────────────────────────────────────
        cv2.rectangle(img, ip(0.20, 0.55), ip(0.37, 0.73), 255, -1)

        # ── Right window (square) ─────────────────────────────────────
        cv2.rectangle(img, ip(0.63, 0.55), ip(0.80, 0.73), 255, -1)

        return img   # 0 = house (black), 255 = background/openings (white)

    # ------------------------------------------------------------------
    def _synthetic_dog(self) -> np.ndarray:
        """
        Hand-crafted side-view dog silhouette using OpenCV drawing primitives.
        Coordinate system: image origin at top-left.
        All measurements are fractions of (w, h).
        """
        w, h = self.output_size
        # Start with white background
        img = np.full((h, w), 255, dtype=np.uint8)

        def ip(fx, fy):   # fractional -> integer pixel
            return (int(fx * w), int(fy * h))

        def ia(fx, fy):   # fractional axes
            return (int(fx * w), int(fy * h))

        # ── Torso (wide ellipse) ──────────────────────────────────────
        cv2.ellipse(img, ip(0.48, 0.54), ia(0.27, 0.17), 0, 0, 360, 0, -1)

        # ── Neck (filled quad connecting body to head) ────────────────
        neck = np.array([ip(0.68, 0.41), ip(0.76, 0.33),
                         ip(0.80, 0.46), ip(0.72, 0.50)], dtype=np.int32)
        cv2.fillPoly(img, [neck], 0)

        # ── Head (circle) ─────────────────────────────────────────────
        cv2.circle(img, ip(0.78, 0.37), int(0.12 * w), 0, -1)

        # ── Snout (ellipse protruding right) ─────────────────────────
        cv2.ellipse(img, ip(0.91, 0.42), ia(0.055, 0.040), 0, 0, 360, 0, -1)

        # ── Ear (triangular flap) ─────────────────────────────────────
        ear = np.array([ip(0.76, 0.26), ip(0.86, 0.21),
                        ip(0.84, 0.35)], dtype=np.int32)
        cv2.fillPoly(img, [ear], 0)

        # ── Tail (thin ellipse, tilted up at rear) ────────────────────
        cv2.ellipse(img, ip(0.19, 0.37), ia(0.035, 0.15), -25, 0, 360, 0, -1)

        # ── Four legs (rectangles) ────────────────────────────────────
        lw = int(0.055 * w)
        for lx_f in (0.37, 0.45):   # hind legs
            lx = int(lx_f * w)
            cv2.rectangle(img, (lx - lw // 2, int(0.65 * h)),
                          (lx + lw // 2, int(0.87 * h)), 0, -1)
        for lx_f in (0.61, 0.70):   # front legs
            lx = int(lx_f * w)
            cv2.rectangle(img, (lx - lw // 2, int(0.65 * h)),
                          (lx + lw // 2, int(0.87 * h)), 0, -1)

        # ── Paws (small rounded caps) ─────────────────────────────────
        for lx_f in (0.37, 0.45, 0.61, 0.70):
            cv2.ellipse(img, ip(lx_f, 0.87), ia(0.035, 0.025), 0, 0, 360, 0, -1)

        return img  # 0 = dog (black), 255 = background (white)


# ---------------------------------------------------------------------------
# Step 2 — Contour extraction & simplification
# ---------------------------------------------------------------------------

class ContourExtractor:
    """
    Extracts the largest outer contour of the dog silhouette and simplifies it
    to a manageable polygon using the Douglas-Peucker algorithm.

    Parameters
    ----------
    epsilon_factor : float
        Douglas-Peucker epsilon as a fraction of the contour's arc length.
        Larger -> fewer vertices, less accurate.  0.005–0.015 works well.
    min_area_ratio : float
        Discard contours whose area is < this fraction of the image area.
        Guards against noise / JPEG artefacts.
    morph_close_px : int
        Morphological closing kernel radius to fill tiny holes before contouring.
    """

    def __init__(
        self,
        epsilon_factor: float = 0.008,
        min_area_ratio: float = 0.015,
        morph_close_px: int = 5,
    ):
        self.epsilon_factor = epsilon_factor
        self.min_area_ratio = min_area_ratio
        self.morph_close_px = morph_close_px

    # ------------------------------------------------------------------
    def extract(
        self,
        binary_image: np.ndarray,
    ) -> Tuple[Optional[Polygon], np.ndarray]:
        """
        Extract + simplify the outer contour.

        Parameters
        ----------
        binary_image : np.ndarray (uint8)
            0 = silhouette (foreground), 255 = background.
            (i.e. the convention used by SilhouetteGenerator)

        Returns
        -------
        polygon : Shapely Polygon (Y axis flipped to math convention) or None
        simplified_contour : raw OpenCV contour array (N,1,2 int32)
        """
        img = binary_image.astype(np.uint8)
        h, w = img.shape[:2]

        # ── 1. Threshold to strict binary (foreground = 255) ─────────
        _, fg = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

        # ── 2. Morphological closing: fill micro-holes ────────────────
        if self.morph_close_px > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.morph_close_px * 2 + 1, self.morph_close_px * 2 + 1),
            )
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)

        # ── 3. Find outer contours ────────────────────────────────────
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.warning("No contours found in silhouette.")
            return None, np.array([])

        # ── 4. Filter noise & pick the largest ───────────────────────
        min_area = h * w * self.min_area_ratio
        valid = [c for c in contours if cv2.contourArea(c) >= min_area]
        if not valid:
            logger.warning("All contours below min-area threshold (%.1f px²).", min_area)
            return None, np.array([])

        largest = max(valid, key=cv2.contourArea)

        # ── 5. Douglas-Peucker simplification ─────────────────────────
        arc = cv2.arcLength(largest, closed=True)
        eps = self.epsilon_factor * arc
        simplified = cv2.approxPolyDP(largest, eps, closed=True)

        logger.info(
            "Contour: %d -> %d vertices (epsilon=%.2f, arc=%.1f)",
            len(largest), len(simplified), eps, arc,
        )

        if len(simplified) < 3:
            logger.warning("Simplified contour has < 3 vertices — using original.")
            simplified = largest

        # ── 6. Build Shapely polygon (flip Y: OpenCV origin is top-left) ──
        polygon = polygon_from_contour(simplified)
        if polygon is None or polygon.area < min_area:
            logger.warning("Contour->polygon failed; retrying with un-simplified contour.")
            polygon = polygon_from_contour(largest)

        if polygon is not None:
            polygon = self._flip_y(polygon, h)

        return polygon, simplified

    # ------------------------------------------------------------------
    @staticmethod
    def _flip_y(polygon: Polygon, image_height: int) -> Polygon:
        """OpenCV puts y=0 at the top; flip so y=0 is at the bottom (math convention)."""
        coords = [(x, image_height - y) for x, y in polygon.exterior.coords]
        return _as_valid(Polygon(coords))

    # ------------------------------------------------------------------
    def visualize(
        self,
        binary_image: np.ndarray,
        contour: np.ndarray,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Draw the extracted + simplified contour on a colour copy of the image.
        Green line = polygon boundary, red dots = vertices.
        """
        vis = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis, [contour], -1, (0, 200, 0), 2)
        for pt in contour:
            cv2.circle(vis, tuple(pt[0]), 5, (0, 0, 220), -1)

        if output_path:
            cv2.imwrite(output_path, vis)
            logger.info("Contour debug image saved -> %s", output_path)

        return vis
