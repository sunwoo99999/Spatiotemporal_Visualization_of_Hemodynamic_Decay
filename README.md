# Spatiotemporal Visualization of Hemodynamic Decay in the Visual Cortex

A real-time 3D visualization system that renders how neural signal propagation degrades along the ventral visual stream (V1 → V2 → V3 → hV4) in subjects with Type 2 Diabetes Mellitus (T2DM) and aging, compared to healthy young controls.

---

## What This Project Does

### 1. HRF Parameter Extraction (`dataset/`)

Parses subject-specific fMRI `.mat` files (N = 33) using MATLAB-fitted **7-parameter Double-Gamma HRF models**. For each subject and each visual ROI (V1–V4), the pipeline extracts:

| Parameter        | Meaning                                    |
| ---------------- | ------------------------------------------ |
| `p1`, `q1`, `a1` | Positive BOLD lobe shape and amplitude     |
| `p2`, `q2`, `a2` | Undershoot shape and amplitude             |
| `c` / `dt`       | Hemodynamic offset — the propagation delay |

Clinical groups:

- **Young Controls** — SUB701–SUB722 (n = 11)
- **T2DM / Older** — remaining subjects (n = 22)

Outputs a single validated JSON file (`viz/brain_data.json`) containing group statistics, per-ROI parameters, MNI coordinates, and a brain surface mesh.

---

### 2. Brain Surface Mesh Extraction (`dataset/extract_brain_mesh.py`)

Loads a real ASL-fMRI NIfTI volume, applies a signal-percentile brain mask, runs **Marching Cubes** to produce a cortical surface mesh, and exports:

- `mesh_verts` + `mesh_normals` — surface geometry with proper normals for silhouette rendering
- `mesh_faces` — triangle indices enabling solid-mesh rendering in WebGL
- 60 k interior voxel point cloud for volumetric context

ROI positions follow the **Benson 2014 retinotopic atlas** (Wandell & Winawer 2011; Larsson & Heeger 2006) for anatomically accurate placement of V1–hV4.

---

### 3. Real-Time 3D WebGL Renderer (`viz/glass_brain.html`)

A single-file, zero-dependency browser application that:

- Renders the brain as a **glass-brain mesh** (backface-extrusion outline silhouette) that updates automatically as the camera rotates
- Animates **sliding signal wavefronts** along subject-specific fiber paths (Bézier curves, Force-Directed Edge Bundling style), with wavefront width and brightness driven by the Double-Gamma HRF curve in real time
- Encodes group differences visually:
  - **Young group** — cold cyan signals, tight wavefronts
  - **T2DM / Old group** — warm amber signals, spatially diffused wavefronts reflecting higher variance
- Displays per-ROI `t_peak`, `σ`, and Δ vs. Young in a live analytics panel
- Plots all four ROI HRF waveforms simultaneously in a mini chart synchronized to the simulation clock
- Supports interactive scrubbing, speed control, and PNG export

---

## Data Flow

```
SUB7xx_hrf.mat  →  extract_hrf_json.py  →  hrf_master_data.json
ASL .nii        →  extract_brain_mesh.py →  brain_data.json
                                                    ↓
                                         glass_brain.html  (WebGL)
```

---

## Key Visualized Findings

- **Sequential delay cascade**: `t_peak` decreases monotonically V1 → V4 in both groups, consistent with feedforward propagation
- **T2DM delay shift**: each ROI shows an earlier but lower-amplitude peak, suggesting microvascular disruption shortening the BOLD response
- **V4 instability**: T2DM group V4 standard deviation (σ ≈ 2.32 s) is 4× that of Young Controls (σ ≈ 0.55 s), rendered as diffused wavefront jitter

---

## Stack

| Layer           | Technology                                                   |
| --------------- | ------------------------------------------------------------ |
| Data extraction | Python — `nibabel`, `scikit-image` (Marching Cubes), `scipy` |
| HRF fitting     | MATLAB (`step10_fitting.m`)                                  |
| Renderer        | Three.js r128, WebGL 2, custom GLSL shaders                  |
| Capture         | Puppeteer (headless Chromium)                                |

---

## Running Locally

```bash
# 1. Regenerate brain_data.json (requires fMRI data in data_bak/)
cd dataset
python extract_brain_mesh.py

# 2. Open the visualization (needs a local HTTP server — CORS)
cd viz
npx serve .
# or any static file server, then open http://localhost:PORT/glass_brain.html

# 3. Capture hero images
cd _screenshot
npm install
node capture_t2dm.js
```

---

## File Structure

```
dataset/
  extract_brain_mesh.py   — NIfTI → brain mesh + ROI JSON
  extract_hrf_json.py     — .mat → HRF parameter JSON
  step10_fitting.m        — MATLAB Double-Gamma fitting
  SUB*_hrf.mat            — per-subject fitted HRF parameters

viz/
  glass_brain.html        — main WebGL visualization
  comparison.html         — side-by-side group comparison view
  brain_data.json         — precomputed scene data (mesh + ROIs + fibers)
  hero_young.png          — Young Controls render
  hero_t2dm.png           — T2DM / Old render

_screenshot/
  capture_t2dm.js         — Puppeteer automated capture script
```
