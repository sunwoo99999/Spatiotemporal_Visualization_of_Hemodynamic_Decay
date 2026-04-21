"""
extract_hrf_json.py
====================
PRD Step 1: 33개 SUB*_hrf.mat 파일에서 7-parameter Double-Gamma HRF를
            배치 추출하여 hrf_master_data.json 생성.

모델 (step10_fitting.m 과 동일):
    h(t) = a1 * gampdf(t; p1, q1) - a2 * gampdf(t; p2, q2) + c
    파라미터 순서: [p1, q1, p2, q2, a1, a2, c]

    ※ PRD 명세의 'dt'는 이 코드베이스에서는 'c'(baseline offset)에 해당합니다.
       실제 데이터에 time-shift dt 파라미터는 없습니다.
"""

import os
import json
import datetime
import warnings
import numpy as np
from scipy.io import loadmat
from scipy.stats import gamma as sp_gamma
from scipy.optimize import curve_fit

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_JSON  = os.path.join(DATASET_DIR, "hrf_master_data.json")

# PRD 그룹 정의 (step10_fitting.m 기준)
YOUNG_IDS = {"701","702","703","705","709","712","714","715","716","718","720"}

# 생리학적 유효 범위 (validation)
PHYSIO_BOUNDS = {
    "p1": (4.0,  7.0),
    "q1": (0.5,  1.5),
    "p2": (4.0, 16.0),
    "q2": (0.5,  2.5),
    "a1": (10.0, 80.0),
    "a2": (8.0,  50.0),
    "c":  (-0.2,  0.5),
}

# ─────────────────────────────────────────────
# HRF 모델 & 피팅 (step10_fitting.m → Python 번역)
# ─────────────────────────────────────────────
def gampdf(t, a, b):
    """MATLAB gampdf(t, a, b) = gamma(shape=a, scale=b).pdf(t)"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp_gamma.pdf(t, a=a, scale=b)


def hrf_model(params, t):
    p1, q1, p2, q2, a1, a2, c = params
    return a1 * gampdf(t, p1, q1) - a2 * gampdf(t, p2, q2) + c


def fit_hrf(y):
    """
    curve_fit (TRF, bounds 지원) — Levenberg-Marquardt 계열, 빠르고 안정적.
    Returns: (best_params, success_flag)
    """
    t = np.arange(0, 25, 2, dtype=float)

    lb = [4,   0.5, 4,   0.5, 10,  8,   -0.2]
    ub = [7,   1.5, 16,  2.5, 80, 50,    0.5 ]
    c0 = float(np.clip(y[0], -0.2, 0.5))
    p0 = [6, 1.0, 12, 1.2, 30, 15, c0]

    def model(t, p1, q1, p2, q2, a1, a2, c):
        return hrf_model([p1, q1, p2, q2, a1, a2, c], t)

    best_params = None
    best_cost   = np.inf
    init_candidates = [
        p0,
        [5, 1.0, 10, 1.0, 25, 12, 0.0],
        [6, 1.2, 14, 1.5, 35, 18, 0.0],
    ]
    for p0_candidate in init_candidates:
        p0_c = np.clip(p0_candidate, lb, ub)
        try:
            popt, _ = curve_fit(model, t, y, p0=p0_c, bounds=(lb, ub),
                                method="trf", maxfev=5000,
                                ftol=1e-9, xtol=1e-9)
            cost = float(np.sum((model(t, *popt) - y) ** 2))
            if cost < best_cost:
                best_cost   = cost
                best_params = popt
        except Exception:
            continue

    if best_params is None:
        # fallback: 초기값 그대로 반환
        best_params = np.array(p0, dtype=float)
        return best_params, False

    success = best_cost < 1.0
    return best_params, success


# ─────────────────────────────────────────────
# 파생 지표 계산
# ─────────────────────────────────────────────
def compute_derived(params):
    t_fine = np.arange(0, 24.01, 0.01)
    y_fine = hrf_model(params, t_fine)

    # 양의 피크
    from scipy.signal import find_peaks
    pos_peaks, _ = find_peaks(y_fine)
    neg_peaks, _ = find_peaks(-y_fine)

    t_peak = fwhm = t_trough = peak_amp = trough_amp = None
    if len(pos_peaks) > 0:
        # t_peak < 4 이면 다음 피크 사용 (printHRF 로직)
        idx = pos_peaks[0]
        if t_fine[idx] < 4 and len(pos_peaks) > 1:
            idx = pos_peaks[1]
        peak_amp = float(y_fine[idx])
        t_peak   = float(t_fine[idx])
        above    = y_fine > peak_amp / 2
        fwhm     = float(np.sum(above) * 0.01)

    if len(neg_peaks) > 0:
        idx2 = neg_peaks[0]
        if -y_fine[idx2] < 6 and len(neg_peaks) > 1:
            idx2 = neg_peaks[1]
        trough_amp = float(-y_fine[idx2])
        t_trough   = float(t_fine[idx2])

    return {
        "t_peak_s":    round(t_peak,   3) if t_peak   is not None else None,
        "peak_amp":    round(peak_amp,  4) if peak_amp is not None else None,
        "fwhm_s":      round(fwhm,      3) if fwhm     is not None else None,
        "trough_amp":  round(trough_amp,4) if trough_amp is not None else None,
        "t_trough_s":  round(t_trough,  3) if t_trough is not None else None,
    }


# ─────────────────────────────────────────────
# .mat 로더 (step10_fitting.m getV1/V2/V3 재현)
# ─────────────────────────────────────────────
def load_roi(mat_path):
    """
    Returns dict: {"V1": array(13,), "V2": ..., "V3": ..., "All": ...}
    """
    mat = loadmat(mat_path, squeeze_me=False)
    hrf = mat["hrf_ROI"]
    data = hrf["data"][0, 0]   # shape: (n_roi, n_hemi, n_timepoints)
    cnt  = hrf["cnt"][0,  0]   # shape: (n_roi, n_hemi)

    def weighted_avg(row_indices):
        num = np.zeros(data.shape[2])
        denom = 0.0
        for r in row_indices:
            for h in range(2):
                w = float(cnt[r, h])
                num   += data[r, h, :] * w
                denom += w
        return (num / denom).astype(float) if denom > 0 else num

    V1  = weighted_avg([0, 1])
    V2  = weighted_avg([2, 3])
    V3  = weighted_avg([4, 5])
    # V4: row 6 (higher ventral stream, single eccentricity band)
    V4  = weighted_avg([6])
    All = hrf["all"][0, 0].flatten().astype(float)

    return {"V1": V1, "V2": V2, "V3": V3, "V4": V4, "All": All}


# ─────────────────────────────────────────────
# 검증
# ─────────────────────────────────────────────
def validate_params(params, param_names):
    flags = {}
    for name, val in zip(param_names, params):
        lo, hi = PHYSIO_BOUNDS[name]
        flags[name] = "OK" if lo <= val <= hi else f"OUT_OF_RANGE({val:.3f})"
    return flags


# ─────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────
def main():
    param_names = ["p1", "q1", "p2", "q2", "a1", "a2", "c"]
    roi_names   = ["V1", "V2", "V3", "V4"]

    mat_files = sorted([
        f for f in os.listdir(DATASET_DIR)
        if f.startswith("SUB") and f.endswith("_hrf.mat")
    ])

    print(f"발견된 파일 수: {len(mat_files)}")

    subjects_data = []
    n_young = 0
    n_other = 0
    n_failed = 0

    # ── Pass 1: 모든 피험자 처리 (V4 fallback 위해 그룹 평균 나중에 채움) ──
    group_sums   = {"Young": {r: None for r in roi_names}, "T2DM_Old": {r: None for r in roi_names}}
    group_counts = {"Young": {r: 0 for r in roi_names},   "T2DM_Old": {r: 0 for r in roi_names}}

    for fname in mat_files:
        sub_id = fname.replace("SUB", "").replace("_hrf.mat", "")
        group  = "Young" if sub_id in YOUNG_IDS else "T2DM_Old"
        if group == "Young":
            n_young += 1
        else:
            n_other += 1

        mat_path = os.path.join(DATASET_DIR, fname)
        print(f"  처리 중: {fname} (Group={group})", end=" ... ", flush=True)

        try:
            rois = load_roi(mat_path)
        except Exception as e:
            print(f"[로드 실패] {e}")
            n_failed += 1
            continue

        sub_entry = {
            "id":    sub_id,
            "group": group,
            "file":  fname,
        }

        all_ok = True
        for roi in roi_names:
            y = rois[roi]
            params, success = fit_hrf(y)
            validation = validate_params(params, param_names)
            derived    = compute_derived(params)

            p = dict(zip(param_names, [round(float(v), 4) for v in params]))
            any_bad = any("OUT_OF_RANGE" in v for v in validation.values())
            if any_bad:
                all_ok = False

            sub_entry[roi] = {
                **p,
                **derived,
                "fit_success":  bool(success),
                "imputed":      False,
                "validation":   validation,
                "raw_hrf":      [round(float(v), 5) for v in y],
            }

            # Accumulate for group-mean fallback
            if success and not any_bad:
                arr = np.array(list(p.values()) + list(derived.values()), dtype=object)
                if group_sums[group][roi] is None:
                    group_sums[group][roi]   = {k: v for k, v in {**p, **derived}.items()}
                    group_counts[group][roi] = 1
                else:
                    for k in p:
                        group_sums[group][roi][k] = group_sums[group][roi][k] + p[k]
                    group_counts[group][roi] += 1

        if not all_ok:
            n_failed += 1
            sub_entry["_flag"] = "PARAM_OUT_OF_RANGE"
            print("[경고: 범위 일탈]")
        else:
            print("OK")

        subjects_data.append(sub_entry)

    # ── Pass 2: V4 fit_success=False 인 항목에 그룹 평균 imputation ──
    group_means = {}
    for grp in ["Young", "T2DM_Old"]:
        group_means[grp] = {}
        for roi in roi_names:
            cnt = group_counts[grp][roi]
            if cnt > 0:
                group_means[grp][roi] = {
                    k: round(v / cnt, 4) if isinstance(v, float) else v
                    for k, v in group_sums[grp][roi].items()
                }

    imputed_count = 0
    for sub in subjects_data:
        grp = sub["group"]
        for roi in ["V4"]:   # V4만 대상 (V1-V3는 이미 충분히 수렴)
            entry = sub.get(roi, {})
            if not entry.get("fit_success", True) or entry.get("imputed", False):
                if grp in group_means and roi in group_means[grp]:
                    mean_vals = group_means[grp][roi]
                    sub[roi].update(mean_vals)
                    sub[roi]["fit_success"] = False
                    sub[roi]["imputed"]     = True
                    imputed_count += 1

    print(f"  V4 보간(imputation) 처리: {imputed_count}건")

    # ─── JSON 조립 ───
    output = {
        "metadata": {
            "generated_at":    datetime.datetime.now().isoformat(),
            "source_dir":      DATASET_DIR,
            "n_subjects":      len(subjects_data),
            "n_young":         n_young,
            "n_t2dm_old":      n_other,
            "n_flagged":       n_failed,
            "parameter_names": param_names,
            "parameter_desc": {
                "p1": "Positive lobe shape (gamma shape)",
                "q1": "Positive lobe scale (gamma scale)",
                "p2": "Undershoot shape (gamma shape)",
                "q2": "Undershoot scale (gamma scale)",
                "a1": "Peak amplitude",
                "a2": "Undershoot amplitude",
                "c":  "Baseline offset (PRD 명세의 dt에 해당 — 실제 모델은 time-shift 없음)",
            },
            "model_equation":  "h(t) = a1*gampdf(t,p1,q1) - a2*gampdf(t,p2,q2) + c",
            "roi_description": {
                "V1": "Primary visual cortex (rows 0-1, retinotopic, early visual)",
                "V2": "Secondary visual cortex (rows 2-3, border V1)",
                "V3": "Tertiary visual cortex (rows 4-5, dorsal/ventral V3)",
                "V4": "Ventral stream V4 (row 6, color/form perception — Ventral What-pathway)",
            },
            "time_vector_s":   list(range(0, 25, 2)),
            "group_A_young":   sorted(YOUNG_IDS),
        },
        "subjects": subjects_data,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n=== 완료 ===")
    print(f"  총 피험자:  {len(subjects_data)}")
    print(f"  Young:      {n_young}")
    print(f"  T2DM/Old:   {n_other}")
    print(f"  플래그 수:  {n_failed}")
    print(f"  출력 파일:  {OUTPUT_JSON}")
    size_kb = os.path.getsize(OUTPUT_JSON) / 1024
    print(f"  파일 크기:  {size_kb:.1f} KB  ({'✓ 500KB 이하' if size_kb < 500 else '✗ 초과'})")


if __name__ == "__main__":
    main()
