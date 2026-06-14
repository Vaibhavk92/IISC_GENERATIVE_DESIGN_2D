"""
phase1_vision.py — Phase 1: Silhouette Generation & Contour Extraction

Three components:
  1. SilhouetteGenerator      – text prompt -> binary (B/W) image
     Attempts SDXL via diffusers; falls back to a synthetic dog polygon.
  2. ImageSilhouetteExtractor – image + prompt -> transparent PNG + binary mask
     Removes background (rembg first, GrabCut fallback), preserves internal
     holes (windows, arches, wheel centres).  Saves a clean RGBA silhouette PNG
     and returns the binary mask used by ContourExtractor.
  3. ContourExtractor         – binary image -> simplified Shapely Polygon
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
    Produces a binary silhouette image (uint8, HxW) for a text prompt via SDXL.

    Raises RuntimeError if the diffusers pipeline cannot be loaded — in that
    case use ImageSilhouetteExtractor with an input image instead.
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
        output_size: Tuple[int, int] = (512, 512),
        num_inference_steps: int = 30,
        guidance_scale: float = 9.0,
    ):
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
        self._ensure_pipeline()
        img = self._from_diffusion(prompt)

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
            raise RuntimeError(
                f"SDXL pipeline failed to load: {exc}. "
                "Install diffusers + torch, or use ImageSilhouetteExtractor "
                "with an input image (set INPUT_IMAGE in main.py)."
            ) from exc

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

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary


# ---------------------------------------------------------------------------
# Step 2 — Image-based silhouette extraction
# ---------------------------------------------------------------------------

class ImageSilhouetteExtractor:
    """
    Extracts a clean silhouette from an existing photograph or illustration.

    Workflow
    --------
    1. Load + resize the input image.
    2. Remove background — tries ``rembg`` first (high quality, no GPU needed),
       falls back to OpenCV GrabCut when rembg is not installed.
    3. Preserve internal holes (windows, arches, wheel centres …) by detecting
       enclosed transparent regions inside the object and keeping them as holes.
    4. Save an RGBA PNG:  object = solid black (alpha=255),
                          background + holes = fully transparent (alpha=0).
    5. Return a binary uint8 mask (0=foreground, 255=background) compatible with
       ``ContourExtractor`` and the rest of the pipeline.

    Parameters
    ----------
    output_size : (width, height) — target resolution for resize.
    min_hole_area : minimum pixel area for a region to be kept as an internal hole
                    rather than filled in as noise.
    """

    def __init__(
        self,
        output_size: Tuple[int, int] = (512, 512),
        min_hole_area: int = 200,
    ):
        self.output_size = output_size
        self.min_hole_area = min_hole_area

    # ------------------------------------------------------------------
    def extract_from_image(
        self,
        image_path: str,
        prompt: str = "",
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Segment the main object from *image_path* and return a pipeline-
        compatible binary mask (0 = foreground, 255 = background/holes).

        Optionally saves a transparent RGBA silhouette PNG to *output_path*.

        Parameters
        ----------
        image_path : str   Path to the input image (JPEG, PNG, WebP, AVIF, HEIC, …).
        prompt     : str   Text hint describing the target object (logged; used
                           by future prompt-guided backends).
        output_path: str   If given, writes the transparent PNG here.

        Returns
        -------
        binary_mask : np.ndarray (uint8, H×W) — 0 = silhouette, 255 = bg/holes
        """
        if prompt:
            logger.info("ImageSilhouetteExtractor: prompt='%s'", prompt)

        img_bgr = self._load_image(image_path)
        img_bgr = cv2.resize(img_bgr, self.output_size, interpolation=cv2.INTER_AREA)
        logger.info("Input image resized to %s", self.output_size)

        # ── 1. Background removal → alpha mask (255=fg, 0=bg) ────────
        alpha = self._remove_background(img_bgr, prompt=prompt)

        # ── 2. Preserve internal holes ────────────────────────────────
        alpha = self._preserve_holes(alpha)

        # ── 3. Post-process: remove isolated noise ────────────────────
        alpha = self._clean_mask(alpha)

        # ── 4. Save transparent PNG ───────────────────────────────────
        if output_path:
            self._save_transparent_png(alpha, output_path)

        # ── 5. Convert to pipeline convention (0=fg, 255=bg) ─────────
        binary_mask = cv2.bitwise_not(alpha)
        return binary_mask

    # ------------------------------------------------------------------
    @staticmethod
    def _load_image(image_path: str) -> np.ndarray:
        """
        Load any image format into a BGR uint8 numpy array.

        cv2.imread handles JPEG/PNG/BMP/TIFF/WebP out of the box.
        For formats it can't decode (AVIF, HEIC, JXL, …) we fall back
        to Pillow, which supports them when the relevant codec is installed
        (e.g. ``pillow-avif-plugin`` or Pillow >= 10 built with libaom).
        """
        img_bgr = cv2.imread(image_path)
        if img_bgr is not None:
            return img_bgr

        # cv2 failed — try Pillow as a format-agnostic fallback
        try:
            from PIL import Image as PILImage
            pil_img = PILImage.open(image_path).convert("RGB")
            img_rgb = np.array(pil_img, dtype=np.uint8)
            return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception as pil_exc:
            raise FileNotFoundError(
                f"Could not load image '{image_path}'. "
                f"cv2.imread returned None; Pillow also failed: {pil_exc}"
            ) from pil_exc

    # ------------------------------------------------------------------
    def _remove_background(self, img_bgr: np.ndarray, prompt: str = "") -> np.ndarray:
        """Return alpha mask: 255=foreground object, 0=background.

        Uses OWL-ViT to locate the object from the prompt, then runs GrabCut
        initialised on that tight bbox.  Falls back to a centre-crop rect if
        OWL-ViT is unavailable or scores too low.
        """
        bbox = self._detect_bbox(img_bgr, prompt) if prompt else None
        return self._grabcut(img_bgr, bbox=bbox)
 
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_bbox(
        img_bgr: np.ndarray,
        prompt: str,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Use OWL-ViT zero-shot object detection to find the bounding box of the
        object described by *prompt*.

        Returns (x1, y1, x2, y2) in pixel coordinates, or None if the model is
        unavailable or no detection exceeds the confidence threshold.

        Optional dependency: ``pip install transformers torch``
        """
        try:
            from transformers import pipeline as hf_pipeline
            from PIL import Image as PILImage

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(img_rgb)

            detector = hf_pipeline(
                "zero-shot-object-detection",
                model="google/owlvit-base-patch32",
            )
            results = detector(pil_img, candidate_labels=[prompt])

            if not results:
                logger.info("OWL-ViT: no detections for prompt '%s'.", prompt)
                return None

            best = max(results, key=lambda r: r["score"])
            logger.info(
                "OWL-ViT detected '%s' (score=%.3f): %s",
                prompt, best["score"], best["box"],
            )

            if best["score"] < 0.05:
                logger.info("OWL-ViT score too low — ignoring detection.")
                return None

            b = best["box"]
            return (int(b["xmin"]), int(b["ymin"]), int(b["xmax"]), int(b["ymax"]))

        except ImportError:
            logger.info("transformers not installed — skipping OWL-ViT detection.")
            return None
        except Exception as exc:
            logger.warning("OWL-ViT detection failed (%s) — skipping.", exc)
            return None

    # ------------------------------------------------------------------
    def _rembg(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Use rembg for high-quality background removal. Returns None on failure."""
        try:
            from rembg import remove
            from PIL import Image as PILImage

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_in = PILImage.fromarray(img_rgb)
            pil_out = remove(pil_in)

            result = np.array(pil_out.convert("RGBA"), dtype=np.uint8)
            alpha_channel = result[:, :, 3]

            _, binary_alpha = cv2.threshold(alpha_channel, 127, 255, cv2.THRESH_BINARY)
            logger.info("rembg background removal succeeded.")
            return binary_alpha
        except ImportError:
            logger.info("rembg not installed — will use GrabCut fallback.")
            return None
        except Exception as exc:
            logger.warning("rembg failed (%s) — falling back to GrabCut.", exc)
            return None

    # ------------------------------------------------------------------
    def _grabcut(
        self,
        img_bgr: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """
        GrabCut segmentation.

        If *bbox* (x1, y1, x2, y2) is provided (from OWL-ViT), it is used
        directly as the initialisation rectangle.  Otherwise falls back to a
        fixed 8%-margin centred rect.
        """
        h, w = img_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            # Small inward pad so the rect doesn't clip the object boundary
            pad_x = max(int(0.02 * w), 2)
            pad_y = max(int(0.02 * h), 2)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            rect = (x1, y1, x2 - x1, y2 - y1)
            logger.info("GrabCut init rect from OWL-ViT bbox: %s", rect)
        else:
            margin_x = max(int(0.08 * w), 5)
            margin_y = max(int(0.08 * h), 5)
            rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
            logger.info("GrabCut init rect from centre-crop fallback: %s", rect)

        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 8,
                    cv2.GC_INIT_WITH_RECT)

        fg_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, k)
        logger.info("GrabCut background removal succeeded.")
        return fg_mask

    # ------------------------------------------------------------------
    def _preserve_holes(self, alpha: np.ndarray) -> np.ndarray:
        """
        Detect enclosed transparent regions inside the object (holes) and keep
        them transparent.  Regions touching the image border are background and
        must stay transparent; enclosed pockets that were incorrectly filled by
        a closing step are restored here.

        Uses connected-components on the *inverted* mask to separate background
        (border-connected components) from interior holes.
        """
        inv = cv2.bitwise_not(alpha)   # 255 = transparent areas (bg + potential holes)

        # Label all transparent blobs
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            inv, connectivity=8
        )

        h, w = alpha.shape
        border_labels: set = set()

        # Any blob touching the image border belongs to the background
        for row in (0, h - 1):
            for c in range(w):
                border_labels.add(int(labels[row, c]))
        for col in (0, w - 1):
            for r in range(h):
                border_labels.add(int(labels[r, col]))

        # Interior blobs not touching border are genuine holes — keep transparent
        result = alpha.copy()
        for lbl in range(1, n_labels):
            if lbl in border_labels:
                continue
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area >= self.min_hole_area:
                result[labels == lbl] = 0   # restore as transparent (hole)
            # Small blobs below min_hole_area are noise — leave as opaque (filled)
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _clean_mask(alpha: np.ndarray) -> np.ndarray:
        """Remove small isolated foreground noise islands."""
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            alpha, connectivity=8
        )
        if n_labels <= 1:
            return alpha

        # Keep only the largest foreground blob (the main object)
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_lbl = int(np.argmax(areas)) + 1

        clean = np.zeros_like(alpha)
        clean[labels == largest_lbl] = 255
        return clean

    # ------------------------------------------------------------------
    @staticmethod
    def _save_transparent_png(alpha: np.ndarray, output_path: str) -> None:
        """
        Write a solid-black silhouette with transparent background as RGBA PNG.
        Object pixels: (0, 0, 0, 255).  Background / hole pixels: (0, 0, 0, 0).
        """
        from PIL import Image as PILImage

        h, w = alpha.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 3] = alpha          # alpha channel only; RGB stay 0 (black)

        PILImage.fromarray(rgba, mode="RGBA").save(output_path)
        logger.info("Transparent silhouette PNG saved -> %s", output_path)


# ---------------------------------------------------------------------------
# Step 3 — Contour extraction & simplification
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
