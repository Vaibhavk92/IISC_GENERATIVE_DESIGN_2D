import os
import sys
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

import streamlit as st
from PIL import Image

from geometry_utils import normalize_polygon
from main import (
    build_inventory, load_inventory_from_folder,
    plot_score_curve, save_per_piece_cuts, visualize, visualize_all,
)
from models import EnvironmentBounds
from phase1_vision import (
    ContourExtractor, ImageSilhouetteExtractor,
    SDXLTurboGenerator, SilhouetteGenerator,
)
from phase2_multiscale import MultiScaleOptimizer
from phase4_fitness import BestLayoutSelector, FitnessEvaluator


@st.cache_resource(show_spinner="Loading SDXL-Turbo model (first run only)…")
def _load_turbo_pipeline(steps: int):
    gen = SDXLTurboGenerator(output_size=(512, 512), num_inference_steps=steps)
    gen._ensure_pipeline()
    return gen


@st.cache_resource(show_spinner="Loading SDXL (full) model…")
def _load_sdxl_pipeline():
    return SilhouetteGenerator(output_size=(512, 512))

# ---------------------------------------------------------------------------
st.set_page_config(page_title="Generative Design 2D", layout="wide", page_icon="🏠")
st.title("Generative Design 2D")
st.caption("Optimal scrap material placement on a target silhouette.")

BOUNDS        = EnvironmentBounds(width=600, height=500)
FITNESS_W     = (0.05, 0.04, 0.85, 0.06)
LAYOUT_KWARGS = {"n_structural_zones": 6,
                 "rotation_candidates": [0, 90, 180, 270],
                 "min_remaining_area": 50.0}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")

    mode = st.radio("Input mode",
                    ["Existing image", "SDXL-Turbo", "SDXL (full)"])

    uploaded_file = None
    steps = 2

    if mode == "Existing image":
        uploaded_file = st.file_uploader(
            "Upload image", type=["jpg", "jpeg", "png", "webp", "avif"])
        prompt = st.text_input("Object prompt (for detection)", value="house")
    elif mode == "SDXL-Turbo":
        prompt = st.text_input("Text prompt", value="house")
        steps  = st.slider("Inference steps", 1, 4, 2)
    else:
        prompt = st.text_input("Text prompt", value="house")

    st.divider()
    st.subheader("Scale sweep")
    scale_min  = st.slider("Min scale",  0.3, 0.9, 0.5, 0.1)
    scale_max  = st.slider("Max scale",  0.5, 1.0, 1.0, 0.1)
    scale_step = st.select_slider("Step", options=[0.05, 0.10, 0.20], value=0.10)

    st.divider()
    run_btn = st.button("Run Pipeline", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(mode, prompt, uploaded_file, steps, scale_min, scale_max, scale_step):
    status = st.status("Running pipeline…", expanded=True)

    # ── Phase 1 ──────────────────────────────────────────────────────────
    status.write("Phase 1 — silhouette extraction…")

    if mode == "Existing image":
        if uploaded_file is None:
            st.error("Please upload an image.")
            return None
        suffix = Path(uploaded_file.name).suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(uploaded_file.read())
            img_path = f.name
        try:
            ext = ImageSilhouetteExtractor(output_size=(512, 512))
            silhouette = ext.extract_from_image(
                image_path=img_path, prompt=prompt,
                output_path=str(BASE_DIR / "silhouette.png"))
        finally:
            os.unlink(img_path)

    elif mode == "SDXL-Turbo":
        gen = _load_turbo_pipeline(steps)
        silhouette = gen.generate(
            prompt=prompt,
            output_path=str(BASE_DIR / "silhouette.png"),
            raw_output_path=str(BASE_DIR / "generated_raw.png"))
    else:
        gen = _load_sdxl_pipeline()
        silhouette = gen.generate(
            prompt=prompt, output_path=str(BASE_DIR / "silhouette.png"))

    status.write("Phase 1 — contour extraction…")
    c_ext = ContourExtractor(epsilon_factor=0.008, min_area_ratio=0.015)
    target_poly, raw_contour = c_ext.extract(silhouette)
    if target_poly is None:
        st.error("Contour extraction failed — try a different image or prompt.")
        return None
    c_ext.visualize(silhouette, raw_contour,
                    output_path=str(BASE_DIR / "contour_debug.png"))
    target_norm = normalize_polygon(target_poly, BOUNDS)

    # ── Phase 2-4 ─────────────────────────────────────────────────────────
    status.write("Phase 2 — multi-scale optimisation…")
    inventory = load_inventory_from_folder(bounds=BOUNDS) or build_inventory()

    optimizer = MultiScaleOptimizer(
        inventory=inventory, bounds=BOUNDS,
        scale_min=scale_min, scale_max=scale_max, scale_step=scale_step,
        fitness_weights=FITNESS_W, layout_kwargs=LAYOUT_KWARGS)
    best = optimizer.run(target_norm)

    status.write("Phase 4 — scoring & outputs…")
    selector = BestLayoutSelector()
    selector.add_all(optimizer.all_results)

    visualize(best,  out_path=str(BASE_DIR / "output_layout.png"))
    visualize_all(optimizer.all_results, best,
                  out_path=str(BASE_DIR / "all_scales_layout.png"))
    plot_score_curve(selector, out_path=str(BASE_DIR / "score_curve.png"))
    save_per_piece_cuts(best, out_dir=str(BASE_DIR / "cuts_output"))

    status.update(label="Pipeline complete!", state="complete")
    return best, selector, optimizer.all_results

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if run_btn:
    result = run_pipeline(mode, prompt, uploaded_file, steps,
                          scale_min, scale_max, scale_step)
    if result:
        best, selector, all_results = result
        evaluator = FitnessEvaluator(weights=FITNESS_W)

        st.success(
            f"Best scale: **{best.scale_factor:.0%}** | "
            f"Score: **{best.score:.4f}** | "
            f"Coverage: **{best.layout.coverage_ratio*100:.1f}%** | "
            f"Cuts: **{best.layout.total_cuts}**"
        )

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
raw_p    = BASE_DIR / "generated_raw.png"
sil_p    = BASE_DIR / "silhouette.png"
cont_p   = BASE_DIR / "contour_debug.png"
layout_p = BASE_DIR / "output_layout.png"
all_p    = BASE_DIR / "all_scales_layout.png"
score_p  = BASE_DIR / "score_curve.png"

if sil_p.exists():
    st.divider()
    st.subheader("Phase 1 — Silhouette")
    cols = st.columns(3)

    if raw_p.exists():
        with cols[0]:
            st.markdown("**Generated image**")
            st.image(str(raw_p))

    with cols[1]:
        st.markdown("**Silhouette mask**")
        st.image(str(sil_p))

    with cols[2]:
        st.markdown("**Contour debug**")
        if cont_p.exists():
            st.image(str(cont_p))

if layout_p.exists():
    st.divider()
    st.subheader("Phase 4 — Layout")
    st.image(str(layout_p), use_container_width=True)

if all_p.exists():
    st.subheader("All scale layouts")
    st.image(str(all_p), use_container_width=True)

if score_p.exists():
    st.subheader("Score curve")
    st.image(str(score_p))

# ---------------------------------------------------------------------------
# Per-piece cut diagrams (separate expandable section)
# ---------------------------------------------------------------------------
cuts_dir = BASE_DIR / "cuts_output"
cut_pngs = sorted(cuts_dir.glob("*.png")) if cuts_dir.exists() else []

if cut_pngs:
    st.divider()
    st.subheader("Per-piece Cut Diagrams")
    st.caption(f"{len(cut_pngs)} piece(s) — click a piece to expand the diagram and download its cut data.")

    for png in cut_pngs:
        piece_id = png.stem
        json_path = png.with_suffix(".json")

        with st.expander(f"✂ {piece_id}", expanded=False):
            col_img, col_json = st.columns([2, 1])

            with col_img:
                st.image(str(png), use_container_width=True)

            with col_json:
                if json_path.exists():
                    import json as _json
                    data = _json.loads(json_path.read_text(encoding="utf-8"))
                    st.markdown(f"**Rotation:** {data.get('rotation_degrees', '?'):.0f}°")
                    st.markdown(f"**Cuts:** {data.get('num_cuts', '?')}")
                    st.markdown(f"**Cut length:** {data.get('total_cut_length_cm', '?')} cm")
                    st.markdown(f"**Covered:** {data.get('covered_area_cm2', '?')} cm²")
                    st.markdown(f"**Waste:** {data.get('waste_area_cm2', '?')} cm²")

                    st.divider()
                    st.markdown("**Download**")
                    st.download_button(
                        label="cut data (.json)",
                        data=json_path.read_bytes(),
                        file_name=json_path.name,
                        mime="application/json",
                        key=f"dl_{piece_id}",
                    )
