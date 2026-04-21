# Spatiotemporal Visualization of Hemodynamic Decay in the Visual Cortex

Real-time 3D WebGL visualization encoding neurovascular coupling degradation along the ventral visual stream (V1 → V4) in Type 2 Diabetes Mellitus (T2DM) versus healthy young controls, driven by subject-specific HRF parameters from 33 clinical fMRI subjects.

---

## Quick Start - open the visualization

`brain_data.json` is already precomputed and included. No Python or MATLAB needed to view it.

```bash
cd viz
npx serve .
# Open http://localhost:3000/glass_brain.html in Chrome or Edge
```

> **Note:** Must be served via HTTP (not `file://`) due to browser CORS restrictions on JSON loading.

---

## Prerequisites

### To view only

| Requirement    | Version    |
| -------------- | ---------- |
| Node.js        | 18+        |
| Chrome or Edge | any recent |

```bash
npm install -g serve   # one-time global install
```

### To capture hero images (optional)

```bash
cd _screenshot
npm install            # installs puppeteer locally
```

### To regenerate `brain_data.json` from raw fMRI (optional)

Requires the raw NIfTI + `.mat` files in `data_bak/`.

```bash
pip install nibabel scikit-image scipy numpy
cd dataset
python extract_brain_mesh.py
```

---

## Usage

### View the visualization

```bash
cd viz
npx serve .
```

Open `http://localhost:3000/glass_brain.html`.

- Toggle **YOUNG (n=11)** / **T2DM / OLD (n=22)** buttons to switch groups
- Drag to rotate, scroll to zoom
- Use the timeline scrubber or **PEAK DEMO** to animate the HRF wavefront

### Capture images

```bash
cd _screenshot
node capture_young.js   # → viz/hero_young.png
node capture_t2dm.js    # → viz/hero_t2dm.png
```

---

## What the visualization encodes

Three simultaneous visual channels map real HRF parameters to perception:

| Channel             | Young Controls          | T2DM / Old                 |
| ------------------- | ----------------------- | -------------------------- |
| **Wavefront width** | Narrow — tight coupling | 3× wider — diffuse         |
| **Glow radius**     | Compact (low σ)         | Expanded ∝ inter-subject σ |
| **Luminance**       | Reference amplitude     | Dimmer ∝ BOLD attenuation  |

Key findings from the data:

- V4 `t_peak` compresses by **−505 ms** in T2DM (earlier, shorter BOLD response)
- V1 inter-subject variance elevated **+57%** in T2DM
- V4 absolute σ is **4.3×** the V1–V3 mean — rendered as the largest, most jittered glow

---

## Data Flow

```
SUB7xx_hrf.mat       → extract_hrf_json.py   → hrf_master_data.json  ─┐
ASL .nii (data_bak/) → extract_brain_mesh.py → brain_data.json  ←──────┘
                                                       ↓
                                            glass_brain.html  (WebGL)
```

`brain_data.json` (precomputed, ~1.8 MB) contains the brain mesh, ROI positions, fiber paths, and HRF statistics. The renderer reads this file directly — no server-side computation.

---

## Stack

| Layer           | Technology                                                   |
| --------------- | ------------------------------------------------------------ |
| HRF fitting     | MATLAB — 7-parameter Double-Gamma nonlinear least squares    |
| Mesh extraction | Python — `nibabel`, `scikit-image` (Marching Cubes), `scipy` |
| Renderer        | Three.js r128, WebGL 2, custom GLSL shaders                  |
| Image capture   | Puppeteer (non-headless Chromium, real GPU)                  |

---

## File Structure

```
viz/
  glass_brain.html          — main WebGL visualization (single file, zero CDN deps)
  comparison.html           — side-by-side group comparison view
  brain_data.json           — precomputed scene data (mesh + ROIs + fibers + HRF stats)
  hero_young.png/jpg        — Young Controls render
  hero_t2dm.png             — T2DM / Old render
  representative_image.jpg  — SIGGRAPH submission image

dataset/
  extract_brain_mesh.py     — NIfTI → cortical mesh + point cloud + ROI JSON
  extract_hrf_json.py       — .mat → HRF parameter JSON
  step10_fitting.m          — MATLAB Double-Gamma fitting pipeline
  hrf_master_data.json      — intermediate HRF stats per subject
  SUB*_hrf.mat              — per-subject fitted HRF parameters

_screenshot/
  capture_young.js          — Puppeteer capture → hero_young.png
  capture_t2dm.js           — Puppeteer capture → hero_t2dm.png
  capture_repimg.js         — Puppeteer capture → representative_image.jpg
  package.json              — puppeteer dependency

abstract/
  siggraph2026_src.tex      — SIGGRAPH 2026 Posters extended abstract (LaTeX)
  refs.bib                  — BibTeX references
```
