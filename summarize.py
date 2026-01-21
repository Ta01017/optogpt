#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
summarize_optogpt_dataset.py

Summarize an OptoGPT-style dataset:
- materials list
- material pairs statistics (adjacent pairs + "head pair")
- thickness range per material
- length stats
- overlay plot of many curves (sampled)
- saves summary.json / summary.txt / curves_overlay.png

Assumes:
- Structure_*.pkl: List[List[str]] where token="Material_ThicknessNm"
- Spectrum_*.pkl : List[List[float]] each spec is [R..., T...] (2W)
- meta_*.pkl     : List[dict] (optional; used for type)
"""

import os
import json
import pickle as pkl
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt


# =========================
# Config: CHANGE THESE
# =========================
DATA_DIR = "./dataset/mix_optogpt_style_v2p1_noFP_noShortAR"  # <-- change to your OUT_DIR
OUT_DIR = os.path.join(DATA_DIR, "_summary")
os.makedirs(OUT_DIR, exist_ok=True)

# Wavelength grid (must match your generator)
LAMBDA0 = 0.9
LAMBDA1 = 1.7
STEP_UM = 0.005

# Plot controls (avoid drawing too many)
MAX_CURVES_PER_SPLIT = 800   # per split (train/dev/test). Increase if you want.
RANDOM_SEED = 42

# If you want plot only R or only T:
PLOT_MODE = "R_T"  # "R" or "T" or "R_T"


# =========================
# Helpers
# =========================
def load_pkl(path: str):
    with open(path, "rb") as f:
        return pkl.load(f)

def parse_token(tok: str):
    # token: "Material_ThicknessNm"
    if "_" not in tok:
        return None, None
    m, t = tok.split("_", 1)
    try:
        thk = float(t)
    except Exception:
        return m, None
    return m, thk

def head_pair(tokens: List[str]) -> str:
    mats = []
    for tok in tokens:
        m, th = parse_token(tok)
        if m is None:
            continue
        mats.append(m)
        if len(mats) >= 2:
            break
    if len(mats) < 2:
        return "INVALID"
    return f"{mats[0]}/{mats[1]}"

def adjacent_pairs(tokens: List[str]) -> List[str]:
    mats = []
    for tok in tokens:
        m, th = parse_token(tok)
        if m is None:
            continue
        mats.append(m)
    pairs = []
    for i in range(len(mats) - 1):
        pairs.append(f"{mats[i]}/{mats[i+1]}")
    return pairs

def slice_spec(spec: np.ndarray, mode: str) -> np.ndarray:
    mode = mode.upper()
    if mode == "R_T":
        return spec
    L = spec.shape[0]
    if L % 2 != 0:
        raise ValueError(f"spec length {L} not even; cannot split R/T")
    W = L // 2
    if mode == "R":
        return spec[:W]
    if mode == "T":
        return spec[W:]
    raise ValueError(mode)

def percentile_str(x: np.ndarray, ps=(0, 5, 25, 50, 75, 95, 100)) -> Dict[str, float]:
    out = {}
    for p in ps:
        out[str(p)] = float(np.percentile(x, p))
    return out

def safe_sample_indices(n: int, k: int, rng: np.random.Generator):
    if n <= k:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=k, replace=False)

def plot_overlay(curves: np.ndarray, wavelengths: np.ndarray, title: str, save_path: str):
    """
    curves: [N, W] or [N, 2W] depending on mode already sliced.
    This draws many lines in one plot. Use sampling to keep it readable.
    """
    plt.figure()
    for i in range(curves.shape[0]):
        plt.plot(wavelengths, curves[i])
    plt.xlabel("Wavelength (um)")
    plt.ylabel("R/T")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# =========================
# Main summarization
# =========================
def summarize_split(name: str, struct_path: str, spec_path: str, meta_path: str = None):
    structs = load_pkl(struct_path)
    specs = np.asarray(load_pkl(spec_path), dtype=np.float32)

    metas = None
    if meta_path is not None and os.path.exists(meta_path):
        metas = load_pkl(meta_path)
        if len(metas) != len(structs):
            print(f"[WARN] meta length mismatch for {name}: meta={len(metas)} struct={len(structs)}")
            metas = None

    assert len(structs) == specs.shape[0], f"{name}: len(struct) != len(spec)"
    n = len(structs)

    # spectrum slicing
    specs_s = np.stack([slice_spec(specs[i], PLOT_MODE) for i in range(n)], axis=0)
    dim = specs_s.shape[1]

    # wavelength grid for plotting
    if PLOT_MODE.upper() == "R_T":
        W = dim // 2
        wavelengths = np.linspace(LAMBDA0, LAMBDA1, int(round((LAMBDA1 - LAMBDA0) / STEP_UM)) + 1)
        if len(wavelengths) != W:
            # fallback: infer from dim
            W = dim // 2
            wavelengths = np.linspace(LAMBDA0, LAMBDA1, W)
        # For overlay, plot just R (first W) and T (second W) separately? We'll do two plots later.
    else:
        wavelengths = np.linspace(LAMBDA0, LAMBDA1, int(round((LAMBDA1 - LAMBDA0) / STEP_UM)) + 1)
        if len(wavelengths) != dim:
            wavelengths = np.linspace(LAMBDA0, LAMBDA1, dim)

    # stats containers
    mat_thks = defaultdict(list)      # mat -> [thk...]
    head_pairs = Counter()
    adj_pairs = Counter()
    lens = []
    type_cnt = Counter()

    for i, toks in enumerate(structs):
        lens.append(len(toks))

        # type from meta if available
        if metas is not None and isinstance(metas[i], dict) and "type" in metas[i]:
            type_cnt[metas[i]["type"]] += 1

        head_pairs[head_pair(toks)] += 1
        for p in adjacent_pairs(toks):
            adj_pairs[p] += 1

        for tok in toks:
            m, th = parse_token(tok)
            if m is None or th is None:
                continue
            mat_thks[m].append(th)

    # thickness stats
    mat_stats = {}
    all_thks = []
    for m, arr in mat_thks.items():
        a = np.asarray(arr, dtype=np.float32)
        all_thks.append(a)
        mat_stats[m] = {
            "count": int(a.size),
            "min": float(a.min()),
            "max": float(a.max()),
            "percentiles": percentile_str(a),
        }
    if all_thks:
        all_thks = np.concatenate(all_thks, axis=0)
        global_thk = {
            "count": int(all_thks.size),
            "min": float(all_thks.min()),
            "max": float(all_thks.max()),
            "percentiles": percentile_str(all_thks),
        }
    else:
        global_thk = {}

    # length stats
    lens_np = np.asarray(lens, dtype=np.int64)
    len_stats = {
        "count": int(lens_np.size),
        "min": int(lens_np.min()),
        "mean": float(lens_np.mean()),
        "max": int(lens_np.max()),
        "p50": float(np.percentile(lens_np, 50)),
        "p90": float(np.percentile(lens_np, 90)),
        "p95": float(np.percentile(lens_np, 95)),
    }

    # curve sampling for plotting
    rng = np.random.default_rng(RANDOM_SEED + (0 if name == "train" else 1 if name == "dev" else 2))
    sample_ids = safe_sample_indices(n, MAX_CURVES_PER_SPLIT, rng)
    curves_sample = specs_s[sample_ids]

    # also save sampled curves for later debug
    np.save(os.path.join(OUT_DIR, f"curves_sample_{name}_{PLOT_MODE}.npy"), curves_sample)

    summary = {
        "split": name,
        "num_samples": int(n),
        "spec_dim_after_slice": int(dim),
        "plot_mode": PLOT_MODE,
        "materials": sorted(list(mat_thks.keys())),
        "num_materials": int(len(mat_thks)),
        "type_distribution": dict(type_cnt) if type_cnt else None,
        "length_stats": len_stats,
        "global_thickness": global_thk,
        "thickness_by_material": mat_stats,
        "head_pair_top50": head_pairs.most_common(50),
        "adjacent_pair_top50": adj_pairs.most_common(50),
    }
    return summary, curves_sample, wavelengths


def main():
    # locate files
    splits = ["train", "dev", "test"]
    summaries = {}
    curve_bank = {}  # split -> curves_sample
    wavelengths_ref = None

    for sp in splits:
        struct_path = os.path.join(DATA_DIR, f"Structure_{sp}.pkl")
        spec_path   = os.path.join(DATA_DIR, f"Spectrum_{sp}.pkl")
        meta_path   = os.path.join(DATA_DIR, f"meta_{sp}.pkl")

        if not (os.path.exists(struct_path) and os.path.exists(spec_path)):
            raise FileNotFoundError(f"Missing {sp} files under {DATA_DIR}")

        summary, curves_sample, wavelengths = summarize_split(
            sp, struct_path, spec_path, meta_path=meta_path
        )
        summaries[sp] = summary
        curve_bank[sp] = curves_sample
        if wavelengths_ref is None:
            wavelengths_ref = wavelengths

    # merge global overview
    all_materials = sorted(list(set().union(*[set(summaries[s]["materials"]) for s in splits])))
    all_head_pairs = Counter()
    all_adj_pairs = Counter()
    for sp in splits:
        for k, v in summaries[sp]["head_pair_top50"]:
            all_head_pairs[k] += int(v)
        for k, v in summaries[sp]["adjacent_pair_top50"]:
            all_adj_pairs[k] += int(v)

    overview = {
        "data_dir": DATA_DIR,
        "splits": {sp: {"num_samples": summaries[sp]["num_samples"],
                        "num_materials": summaries[sp]["num_materials"],
                        "length_stats": summaries[sp]["length_stats"],
                        "type_distribution": summaries[sp]["type_distribution"]} for sp in splits},
        "all_materials_union": all_materials,
        "num_materials_union": len(all_materials),
        "top_head_pairs_global_50": all_head_pairs.most_common(50),
        "top_adjacent_pairs_global_50": all_adj_pairs.most_common(50),
        "plot_mode": PLOT_MODE,
        "max_curves_per_split": MAX_CURVES_PER_SPLIT,
    }

    # save json
    out_json = os.path.join(OUT_DIR, "summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"overview": overview, "by_split": summaries}, f, ensure_ascii=False, indent=2)

    # save txt (human-friendly)
    out_txt = os.path.join(OUT_DIR, "summary.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"DATA_DIR: {DATA_DIR}\n")
        f.write(f"PLOT_MODE: {PLOT_MODE}\n")
        f.write(f"MAX_CURVES_PER_SPLIT: {MAX_CURVES_PER_SPLIT}\n\n")

        for sp in splits:
            s = summaries[sp]
            f.write(f"== SPLIT: {sp} ==\n")
            f.write(f"num_samples: {s['num_samples']}\n")
            if s["type_distribution"] is not None:
                f.write(f"type_distribution: {s['type_distribution']}\n")
            f.write(f"length_stats: {s['length_stats']}\n")
            f.write(f"num_materials: {s['num_materials']}\n")
            f.write(f"materials: {s['materials']}\n")
            f.write(f"global_thickness: {s['global_thickness']}\n")
            f.write("head_pair_top10:\n")
            for k, v in s["head_pair_top50"][:10]:
                f.write(f"  {k:20s} {v}\n")
            f.write("adjacent_pair_top10:\n")
            for k, v in s["adjacent_pair_top50"][:10]:
                f.write(f"  {k:20s} {v}\n")
            f.write("\n")

        f.write("== GLOBAL ==\n")
        f.write(f"num_materials_union: {overview['num_materials_union']}\n")
        f.write(f"all_materials_union: {overview['all_materials_union']}\n")
        f.write("top_head_pairs_global_10:\n")
        for k, v in overview["top_head_pairs_global_50"][:10]:
            f.write(f"  {k:20s} {v}\n")
        f.write("top_adjacent_pairs_global_10:\n")
        for k, v in overview["top_adjacent_pairs_global_50"][:10]:
            f.write(f"  {k:20s} {v}\n")

    # =========================
    # Plot overlay curves
    # =========================
    # Merge sampled curves across splits
    curves_all = np.concatenate([curve_bank["train"], curve_bank["dev"], curve_bank["test"]], axis=0)

    if PLOT_MODE.upper() == "R_T":
        dim = curves_all.shape[1]
        if dim % 2 != 0:
            raise ValueError("R_T mode expects even dim (R+T).")
        W = dim // 2
        wl = np.linspace(LAMBDA0, LAMBDA1, W) if (wavelengths_ref is None or len(wavelengths_ref) != W) else wavelengths_ref

        out_r = os.path.join(OUT_DIR, "curves_overlay_R.png")
        out_t = os.path.join(OUT_DIR, "curves_overlay_T.png")
        plot_overlay(curves_all[:, :W], wl, f"Overlay curves (R) | sampled={curves_all.shape[0]}", out_r)
        plot_overlay(curves_all[:, W:], wl, f"Overlay curves (T) | sampled={curves_all.shape[0]}", out_t)
    else:
        wl = wavelengths_ref
        out_png = os.path.join(OUT_DIR, f"curves_overlay_{PLOT_MODE}.png")
        plot_overlay(curves_all, wl, f"Overlay curves ({PLOT_MODE}) | sampled={curves_all.shape[0]}", out_png)

    print("Done.")
    print("Saved:")
    print("  ", out_json)
    print("  ", out_txt)
    print("  ", OUT_DIR)


if __name__ == "__main__":
    main()
