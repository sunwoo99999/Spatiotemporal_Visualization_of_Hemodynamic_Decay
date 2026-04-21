"""
extract_brain_mesh.py
---------------------
Extracts brain surface mesh + voxel point cloud from real ASL NII,
plus V1-V4 + surrounding visual cortex ROI definitions with
fiber-tract connectivity. Outputs viz/brain_data.json.
"""

import nibabel as nib
import numpy as np
from skimage import measure
from scipy.ndimage import binary_fill_holes, binary_erosion, gaussian_filter
import json, os

NII_PATH = r"c:\1_research\19_siggraph\proj\data_bak\ASL\swrdr63real_1222_REST1_LR.nii"
OUT_PATH = r"c:\1_research\19_siggraph\proj\viz\brain_data.json"

SCALE = 0.085   # MNI mm → Three.js units  (90mm → ~7.65 units)

# ── 1. Load NII ────────────────────────────────────────────────────────────
print("Loading NII...")
img   = nib.load(NII_PATH)
data4d = img.get_fdata()           # (91, 109, 91, 63)
affine = img.affine                # MNI affine

# Mean over time → 3D
vol = np.mean(data4d, axis=3)     # (91, 109, 91)

# ── 2. Brain mask via Otsu-like threshold + fill holes ─────────────────────
print("Building brain mask...")
smoothed = gaussian_filter(vol, sigma=1.5)
# Use only voxels with >5% max signal before computing percentile
# This excludes background/skull-baseline from the threshold calculation
sig_mask = smoothed > smoothed.max() * 0.05
flat = smoothed[sig_mask].ravel()
thresh = np.percentile(flat, 70)   # top 30% of signal = cortical grey matter
print(f"  Threshold: {thresh:.4f} (70th pct of signal>5% max)")

mask = smoothed > thresh
# fill internal holes per slice
for z in range(mask.shape[2]):
    mask[:,:,z] = binary_fill_holes(mask[:,:,z])

print(f"  Mask voxels: {mask.sum():,}")

# ── 3. Marching cubes surface ──────────────────────────────────────────────
print("Running marching cubes...")
# Slightly erode for cleaner surface
mask_eroded = binary_erosion(mask, iterations=2)
verts_vox, faces, normals, _ = measure.marching_cubes(
    mask_eroded.astype(float), level=0.5, step_size=2,
    allow_degenerate=False
)
print(f"  Surface: {len(verts_vox):,} verts, {len(faces):,} faces")

# Convert voxel coords → MNI → Three.js
def vox_to_threejs(vox_coords, affine):
    """vox_coords: (N,3), returns (N,3) Three.js coords"""
    ones = np.ones((len(vox_coords), 1))
    hom  = np.hstack([vox_coords, ones])        # (N,4)
    mni  = (affine @ hom.T).T[:, :3]            # (N,3) MNI mm
    # MNI → Three.js: x→x, y→z, z→y  (neurological convention)
    tj = np.column_stack([
        mni[:, 0] * SCALE,
        mni[:, 2] * SCALE,
        mni[:, 1] * SCALE
    ])
    return tj

verts_tj = vox_to_threejs(verts_vox, affine)
print(f"  Three.js bbox: x[{verts_tj[:,0].min():.2f},{verts_tj[:,0].max():.2f}]"
      f" y[{verts_tj[:,1].min():.2f},{verts_tj[:,1].max():.2f}]"
      f" z[{verts_tj[:,2].min():.2f},{verts_tj[:,2].max():.2f}]")

# Transform marching-cubes normals to Three.js space
# normals from skimage are in voxel-index space; apply affine rotation (no translation)
rot3 = affine[:3, :3]  # 3×3 rotation+scale part
normals_mni = (rot3 @ normals.T).T          # (N,3) in MNI directions
norms_len = np.linalg.norm(normals_mni, axis=1, keepdims=True)
norms_len = np.where(norms_len < 1e-10, 1.0, norms_len)
normals_mni /= norms_len                    # unit vectors
# MNI → Three.js axis swap (x→x, y→z, z→y) — same as verts
normals_tj = np.column_stack([normals_mni[:,0],
                               normals_mni[:,2],
                               normals_mni[:,1]])

# Downsample vertices AND normals with same stride
TARGET_MESH_VERTS = 14000
step = max(1, len(verts_tj) // TARGET_MESH_VERTS)
mesh_verts = verts_tj[::step].tolist()
mesh_normals_ds = normals_tj[::step]
print(f"  Downsampled mesh verts: {len(mesh_verts):,}")

# ── 4. Interior voxel point cloud ──────────────────────────────────────────
print("Extracting interior point cloud...")
zi, yi, xi = np.where(mask)
vox_all = np.column_stack([xi, yi, zi]).astype(float)

# Downsample to ~60k points
TARGET_PTS = 60000
idx = np.random.choice(len(vox_all), min(TARGET_PTS, len(vox_all)), replace=False)
vox_sample = vox_all[idx]
pts_tj = vox_to_threejs(vox_sample, affine)

# Compute local "intensity" (normalised signal) for brightness
intensities = smoothed[zi[idx], yi[idx], xi[idx]]
ints_norm = ((intensities - intensities.min()) /
             (intensities.max() - intensities.min() + 1e-9)).tolist()
pts_list = pts_tj.tolist()
print(f"  Point cloud: {len(pts_list):,} points")

# ── 5. ROI definitions (MNI mm, bilateral average for now) ─────────────────
# Visual hierarchy ROIs — MNI coordinates from literature
# Format: [mni_x, mni_y, mni_z]
ROIS_MNI = {
    # ── Benson 2014 / Wandell 2007 retinotopic atlas (MNI152, mm) ──────────
    # V1: calcarine sulcus, foveal representation ~±8 mm lateral
    #     Ref: Benson 2014 Fig.3; Dougherty 2003 J Neurosci
    "V1L":  [ -8, -88,  2],   "V1R":  [  8, -88,  2],
    # V2: dorsal band superior to V1; ventral band inferior – using weighted avg
    #     Ref: Larsson & Heeger 2006; Wandell 2007 Neuron
    "V2L":  [-15, -86, 14],  "V2R":  [ 15, -86, 14],
    # V3: dorsal anterior to V2d; ventral ROI merged
    #     Ref: Hansen 2007; Silver & Kastner 2009
    "V3L":  [-24, -80, 22],  "V3R":  [ 24, -80, 22],
    # hV4 (human V4): ventral occipitotemporal, NOT dorsal
    #     Ref: Wandell 2007 Fig.6; Arcaro 2009; MNI peak ≈ ±33,-68,-12
    "V4L":  [-33, -68,-12],  "V4R":  [ 33, -68,-12],
    # ── Surrounding regions ─────────────────────────────────────────────────
    "LOC_L":[-42, -76,  8],  "LOC_R":[ 42, -76,  8],   # LO1/LO2 Lateral occipital
    "V5_L": [-46, -70,  4],  "V5_R": [ 46, -70,  4],   # MT/V5 Zilles 2004
    "FFA_L":[-36, -52,-20],  "FFA_R":[ 36, -52,-20],   # FFA Kanwisher 1997
    "PPA_L":[-26, -48,-12],  "PPA_R":[ 26, -48,-12],   # PPA Epstein 1998
    "IPS_L":[-26, -58, 52],  "IPS_R":[ 26, -58, 52],   # IPS0 Swisher 2007
    "LGN_L":[-21, -26, -4],  "LGN_R":[ 21, -26, -4],   # LGN Chen 1999
}

def mni_to_threejs(mni):
    x = mni[0] * SCALE
    y = mni[2] * SCALE
    z = mni[1] * SCALE
    return [round(x,3), round(y,3), round(z,3)]

rois = {}
for name, mni in ROIS_MNI.items():
    rois[name] = {
        "pos": mni_to_threejs(mni),
        "mni": mni,
        "group": ("V1" if "V1" in name else
                  "V2" if "V2" in name else
                  "V3" if "V3" in name else
                  "V4" if "V4" in name else
                  "other")
    }

# ── 6. Fiber tract connectivity ────────────────────────────────────────────
# Define directed edges: from → to, with anatomical fibre bundles
# We generate multiple Bezier lines per connection (bundle)
CONNECTIONS = [
    # Thalamo-cortical (LGN → V1)
    ("LGN_L", "V1L"), ("LGN_R", "V1R"),
    # V1 → V2
    ("V1L", "V2L"), ("V1R", "V2R"),
    # V2 → V3
    ("V2L", "V3L"), ("V2R", "V3R"),
    # V3 → V4
    ("V3L", "V4L"), ("V3R", "V4R"),
    # V4 → FFA (object recognition)
    ("V4L", "FFA_L"), ("V4R", "FFA_R"),
    # V1 → LOC (global → local)
    ("V1L", "LOC_L"), ("V1R", "LOC_R"),
    # LOC → V5 (motion)
    ("LOC_L", "V5_L"), ("LOC_R", "V5_R"),
    # V3 → IPS (dorsal stream)
    ("V3L", "IPS_L"), ("V3R", "IPS_R"),
    # FFA → PPA
    ("FFA_L", "PPA_L"), ("FFA_R", "PPA_R"),
    # Cross-hemisphere commissural (V1 ↔)
    ("V1L", "V1R"),
    # V2 cross
    ("V2L", "V2R"),
    # V4 cross
    ("V4L", "V4R"),
]

N_FIBERS_PER_CONN = 8   # number of parallel fiber lines per connection

def bezier_ctrl(p0, p1, spread=0.4):
    """Generate a random Bezier control point between p0 and p1."""
    mid = [(p0[i]+p1[i])/2 for i in range(3)]
    perp = [
        (np.random.rand()-0.5)*spread,
        (np.random.rand()-0.5)*spread,
        (np.random.rand()-0.5)*spread
    ]
    return [mid[i]+perp[i] for i in range(3)]

np.random.seed(42)
fibers = []
for src, dst in CONNECTIONS:
    p0 = rois[src]["pos"]
    p1 = rois[dst]["pos"]
    # Straight-line distance
    dist = np.linalg.norm(np.array(p1)-np.array(p0))
    spread = max(0.15, dist * 0.35)
    for _ in range(N_FIBERS_PER_CONN):
        cp1 = bezier_ctrl(p0, p1, spread)
        cp2 = bezier_ctrl(p0, p1, spread)
        fibers.append({
            "src": src,
            "dst": dst,
            "p0":  [round(v,4) for v in p0],
            "cp1": [round(v,4) for v in cp1],
            "cp2": [round(v,4) for v in cp2],
            "p1":  [round(v,4) for v in p1],
        })

print(f"  Fibers: {len(fibers)} total ({len(CONNECTIONS)} connections × {N_FIBERS_PER_CONN})")

# ── 7. HRF group stats (from hrf_master_data.json) ────────────────────────
# Embed directly for convenience
HRF_STATS = {
    "young": {
        "V1": {"tPeak":7.500,"std":0.548,"peakAmp":2.833},
        "V2": {"tPeak":7.295,"std":0.550,"peakAmp":2.273},
        "V3": {"tPeak":7.049,"std":0.506,"peakAmp":2.095},
        "V4": {"tPeak":6.382,"std":1.963,"peakAmp":2.248},
    },
    "t2dm": {
        "V1": {"tPeak":7.209,"std":0.860,"peakAmp":2.640},
        "V2": {"tPeak":6.966,"std":0.731,"peakAmp":2.359},
        "V3": {"tPeak":6.740,"std":0.784,"peakAmp":2.094},
        "V4": {"tPeak":5.877,"std":2.322,"peakAmp":2.149},
    }
}

# ── 8. Write JSON ──────────────────────────────────────────────────────────
print("Writing brain_data.json ...")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# Flatten points+intensities into interleaved [x,y,z,i, x,y,z,i, ...]
pts_flat = []
for i, (pt, iv) in enumerate(zip(pts_list, ints_norm)):
    pts_flat.extend([round(pt[0],4), round(pt[1],4), round(pt[2],4), round(iv,4)])

out = {
    "meta": {
        "source": "swrdr63real_1222_REST1_LR.nii",
        "space": "MNI152_2mm",
        "scale": SCALE,
        "n_pts": len(pts_list),
        "n_mesh_verts": len(mesh_verts)
    },
    "points": pts_flat,               # [x,y,z,intensity, ...]  flat array
    "mesh_verts": [round(v,3) for sub in mesh_verts for v in sub],  # flat [x,y,z,...]
    "mesh_normals": [round(float(v),4)
                     for row in mesh_normals_ds.tolist()
                     for v in row],   # flat [nx,ny,nz,...] matching mesh_verts
    "mesh_faces": faces.ravel().tolist(),   # flat triangle indices [i0,i1,i2, ...]
    "rois": rois,
    "fibers": fibers,
    "hrf": HRF_STATS,
}

with open(OUT_PATH, "w") as f:
    json.dump(out, f, separators=(',',':'))

size_kb = os.path.getsize(OUT_PATH) / 1024
print(f"Done → {OUT_PATH}  ({size_kb:.0f} KB)")
