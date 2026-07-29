# Generative Design 2D — Turning Scrap Material into Value

## The Industry Problem

Manufacturing units, workshops, and fabrication shops generate large volumes of offcuts and scrap material — plywood, sheet metal, acrylic, fabric — every day. Today, most of this material has **no productive use**: it is stockpiled, sold at scrap rates, or sent to landfill. The material itself is perfectly usable; what's missing is a fast, low-cost way to figure out *what can be built from the odd-shaped pieces at hand*.

## Roadmap — From PoC to Industry Impact

This 2D pipeline is the first iteration. The **next iteration extends the same idea to 3D**: fitting scrap stock to 3D part geometries, generating manufacturable prototypes directly from waste inventory.

The target beneficiaries are **manufacturing companies and MSMEs**, where material cost is a major share of expenses and scrap utilisation is currently near zero. Even modest recovery of scrap into sellable prototypes and products translates into direct cost savings, new revenue from waste, and a measurable sustainability story — at a scale that matters for the MSME sector.

## What This Project Does

This is a **proof of concept (PoC)** for a generative design pipeline that gives scrap material a second life. Instead of designing a product first and then buying fresh material for it, the pipeline works in reverse:

1. **Start from a target design** — an uploaded image or an AI-generated silhouette (SDXL / SDXL-Turbo).
2. **Extract the design outline** — computer vision converts the image into a clean 2D contour.
3. **Fit the scrap to the design** — the available scrap inventory (real, odd-shaped pieces stored as JSON) is placed onto the design across multiple scales using zone-based greedy placement.
4. **Score and select the best layout** — a fitness function balances number of cuts, total cut length, uncovered area, and wasted material, and picks the best scale automatically.
5. **Output ready-to-use cutting plans** — per-piece cut diagrams and JSON files (`cuts_output/`) that show exactly how each scrap piece should be cut.

The result: a workshop can take an idea, feed in its actual scrap inventory, and get back a **design prototype plus cutting instructions** — creating value from material that would otherwise be waste.

## Quick Start

```bash
pip install -r requirements.txt

# Interactive app (recommended)
streamlit run app.py

# Or run the pipeline directly
python main.py
```

AI silhouette generation (SDXL) is optional and needs the extra packages listed at the bottom of `requirements.txt` (GPU recommended). The pipeline works fully with uploaded images without them.

## Pipeline Structure

| Phase | Module | Role |
|-------|--------|------|
| 1 | `phase1_vision.py` | Silhouette generation / image input + contour extraction |
| 2 | `phase2_multiscale.py` | Multi-scale sweep of the target design |
| 3 | `phase3_layout.py` | Zone-area matched greedy placement of scrap pieces |
| 4 | `phase4_fitness.py` | Fitness scoring and best-layout selection |

Scrap inventories live in `inventory_pieces/` and generated cutting plans in `cuts_output/`.


