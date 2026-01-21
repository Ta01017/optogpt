#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
summarize_optogpt_dataset_v2_report.py

Changes vs v1:
1) Ignore test split (only train + dev).
2) Print a report-friendly block:
   - 材料种类（数量+列表）
   - 组合种类（数量+详细列表）
     * head_pair: 首两层材料对
     * adjacent_pair: 相邻层材料对（更全面）
   - 厚度具体取值（全局厚度档位 + 每种材料各自出现过的厚度档位）
3) Keep outputs: summary.json / summary.txt / overlay plots + sampled curves numpy.

Assumes:
- Structure_{train,dev}.pkl: List[List[str]] token="Material_ThicknessNm"
- Spectrum_{train,dev}.pkl : List[List[float]] each spec is [R..., T...] (2W)
- meta_{train,dev}.pkl optional
"""

import os
import json
import pickle as pkl
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import matplotlib.pyplot as plt


# =========================
# Config: CHANGE THESE
# =========================
DATA_DIR = "./dataset/mix_optogpt_style_v2"   # <-- change to your OUT_DIR
OUT_DIR = os.path.join(DATA_DIR, "_summary")
os.makedirs(OUT_DIR, exist_ok=True)

# Wavelength grid (must match generator)
LAMBDA0 = 0.9
LAMBDA1 = 1.7
STEP_UM = 0.005

# Plot controls (avoid drawing too many)
MAX_CURVES_PER_SPLIT = 800   # train and dev each
RANDOM_SEED = 42

# "R_T" plots R and T separately; "R" only plots R; "T" only plots T
PLOT_MODE = "R_T"  # "R" or "T" or "R_T"

# How many pairs / thickness values to print in REPORT BLOCK
REPORT_MAX_LIST = 5000  # large enough; you said need "详细的都要"


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
        thk = int(float(t))
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
    return [f"{mats[i]}/{mats[i+1]}" for i in range(len(mats) - 1)]

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

def safe_sample_indices(n: int, k: int, rng: np.random.Generator):
    if n <= k:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=k, replace=False)

def plot_overlay(curves: np.ndarray, wavelengths: np.ndarray, title: str, save_path: str):
    plt.figure()
    for i in range(curves.shape[0]):
        plt.plot(wavelengths, curves[i])
    plt.xlabel("Wavelength (um)")
    plt.ylabel("R/T")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def sorted_unique_ints(xs: List[int]) -> List[int]:
    return sorted(list(set(int(x) for x in xs)))

def list_to_wrapped_str(items: List[str], sep="，", max_per_line=20) -> str:
    # For Chinese report readability
    lines = []
    for i in range(0, len(items), max_per_line):
        lines.append(sep.join(items[i:i+max_per_line]))
    return "\n".join(lines)


# =========================
# Core summarize
# =========================
def summarize_split(split: str) -> Dict[str, Any]:
    struct_path = os.path.join(DATA_DIR, f"Structure_{split}.pkl")
    spec_path   = os.path.join(DATA_DIR, f"Spectrum_{split}.pkl")
    meta_path   = os.path.join(DATA_DIR, f"meta_{split}.pkl")

    if not (os.path.exists(struct_path) and os.path.exists(spec_path)):
        raise FileNotFoundError(f"Missing files for split={split} in {DATA_DIR}")

    structs = load_pkl(struct_path)
    specs = np.asarray(load_pkl(spec_path), dtype=np.float32)

    metas = None
    if os.path.exists(meta_path):
        try:
            metas = load_pkl(meta_path)
            if len(metas) != len(structs):
                metas = None
        except Exception:
            metas = None

    assert len(structs) == specs.shape[0], f"{split}: len(struct) != len(spec)"
    n = len(structs)

    # Slice specs
    specs_s = np.stack([slice_spec(specs[i], PLOT_MODE) for i in range(n)], axis=0)
    dim = specs_s.shape[1]

    # Wavelengths
    if PLOT_MODE.upper() == "R_T":
        if dim % 2 != 0:
            raise ValueError("R_T mode expects even dim (R+T).")
        W = dim // 2
        wl = np.linspace(LAMBDA0, LAMBDA1, int(round((LAMBDA1 - LAMBDA0) / STEP_UM)) + 1)
        if len(wl) != W:
            wl = np.linspace(LAMBDA0, LAMBDA1, W)
    else:
        wl = np.linspace(LAMBDA0, LAMBDA1, int(round((LAMBDA1 - LAMBDA0) / STEP_UM)) + 1)
        if len(wl) != dim:
            wl = np.linspace(LAMBDA0, LAMBDA1, dim)

    # Stats
    mat_set = set()
    mat_thks = defaultdict(list)     # mat -> thickness values (all occurrences)
    mat_thk_set = defaultdict(set)   # mat -> unique thickness values
    all_thk_set = set()

    head_pair_cnt = Counter()
    adj_pair_cnt = Counter()

    lens = []
    type_cnt = Counter()

    for i, toks in enumerate(structs):
        lens.append(len(toks))

        if metas is not None and isinstance(metas[i], dict) and "type" in metas[i]:
            type_cnt[metas[i]["type"]] += 1

        head_pair_cnt[head_pair(toks)] += 1
        for p in adjacent_pairs(toks):
            adj_pair_cnt[p] += 1

        for tok in toks:
            m, th = parse_token(tok)
            if m is None:
                continue
            mat_set.add(m)
            if th is not None:
                mat_thks[m].append(th)
                mat_thk_set[m].add(th)
                all_thk_set.add(th)

    lens_np = np.asarray(lens, dtype=np.int64)

    # Sample curves for overlay
    rng = np.random.default_rng(RANDOM_SEED + (0 if split == "train" else 1))
    sample_ids = safe_sample_indices(n, MAX_CURVES_PER_SPLIT, rng)
    curves_sample = specs_s[sample_ids]
    np.save(os.path.join(OUT_DIR, f"curves_sample_{split}_{PLOT_MODE}.npy"), curves_sample)

    out = {
        "split": split,
        "num_samples": int(n),
        "spec_dim_after_slice": int(dim),
        "plot_mode": PLOT_MODE,
        "materials": sorted(list(mat_set)),
        "num_materials": int(len(mat_set)),
        "type_distribution": dict(type_cnt) if type_cnt else None,
        "length_stats": {
            "min": int(lens_np.min()),
            "mean": float(lens_np.mean()),
            "max": int(lens_np.max()),
            "p50": float(np.percentile(lens_np, 50)),
            "p90": float(np.percentile(lens_np, 90)),
            "p95": float(np.percentile(lens_np, 95)),
        },
        "thickness_values_global": sorted_unique_ints(list(all_thk_set)),
        "thickness_values_by_material": {m: sorted_unique_ints(list(mat_thk_set[m])) for m in sorted(mat_set)},
        "head_pair_distribution": head_pair_cnt.most_common(),      # full list
        "adjacent_pair_distribution": adj_pair_cnt.most_common(),   # full list
        "_curves_sample_path": os.path.join(OUT_DIR, f"curves_sample_{split}_{PLOT_MODE}.npy"),
    }
    return out, curves_sample, wl


def make_report_block(train_sum: Dict[str, Any], dev_sum: Dict[str, Any],
                      global_materials: List[str],
                      global_head_pairs: List[Tuple[str, int]],
                      global_adj_pairs: List[Tuple[str, int]],
                      global_thk_values: List[int],
                      thk_values_by_mat: Dict[str, List[int]]) -> str:
    """
    A single block you can paste into your report.
    """
    lines = []
    lines.append("【数据集统计汇总（Train+Dev）】")
    lines.append(f"- Split 使用：train + dev（不含 test）")
    lines.append(f"- Train 样本数：{train_sum['num_samples']}；Dev 样本数：{dev_sum['num_samples']}；总计：{train_sum['num_samples'] + dev_sum['num_samples']}")
    lines.append("")

    # Materials
    lines.append(f"1）材料种类：{len(global_materials)}")
    lines.append("材料列表：")
    lines.append(list_to_wrapped_str(global_materials, sep="，", max_per_line=18))
    lines.append("")

    # Pairs (detailed list requested)
    lines.append(f"2）组合种类（材料对）")
    lines.append(f"- Head pair（首两层材料对）种类数：{len(global_head_pairs)}")
    lines.append(f"- Adjacent pair（相邻层材料对）种类数：{len(global_adj_pairs)}")
    lines.append("")
    lines.append("Head pair 详细列表（pair: count）：")
    for k, v in global_head_pairs[:REPORT_MAX_LIST]:
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("Adjacent pair 详细列表（pair: count）：")
    for k, v in global_adj_pairs[:REPORT_MAX_LIST]:
        lines.append(f"  - {k}: {v}")
    lines.append("")

    # Thickness
    lines.append("3）厚度离散取值（Thickness values）")
    lines.append(f"- 全局厚度档位数：{len(global_thk_values)}")
    lines.append(f"- 全局厚度档位（nm）：{global_thk_values}")
    lines.append("")
    lines.append("各材料出现过的厚度档位（nm）：")
    for m in global_materials:
        vals = thk_values_by_mat.get(m, [])
        lines.append(f"  - {m}（{len(vals)}档）：{vals}")
    lines.append("")

    # Length stats
    lines.append("4）序列长度（layer 数）统计")
    lines.append(f"- Train: {train_sum['length_stats']}")
    lines.append(f"- Dev  : {dev_sum['length_stats']}")
    lines.append("")

    # Type distribution if present
    if train_sum.get("type_distribution") is not None or dev_sum.get("type_distribution") is not None:
        lines.append("5）类型分布（若 meta 存在）")
        lines.append(f"- Train: {train_sum.get('type_distribution')}")
        lines.append(f"- Dev  : {dev_sum.get('type_distribution')}")
        lines.append("")

    return "\n".join(lines)


def main():
    # Only train + dev
    splits = ["train", "dev"]

    summaries = {}
    curve_bank = {}
    wavelengths_ref = None

    for sp in splits:
        s, curves_sample, wl = summarize_split(sp)
        summaries[sp] = s
        curve_bank[sp] = curves_sample
        if wavelengths_ref is None:
            wavelengths_ref = wl

    # Global unions across train+dev
    global_materials = sorted(list(set(summaries["train"]["materials"]).union(set(summaries["dev"]["materials"]))))

    # Merge pair counts (use full distributions)
    head_cnt = Counter()
    adj_cnt = Counter()
    for sp in splits:
        for k, v in summaries[sp]["head_pair_distribution"]:
            head_cnt[k] += int(v)
        for k, v in summaries[sp]["adjacent_pair_distribution"]:
            adj_cnt[k] += int(v)

    global_head_pairs = head_cnt.most_common()
    global_adj_pairs = adj_cnt.most_common()

    # Thickness union + per material union
    global_thk_values = sorted(list(set(summaries["train"]["thickness_values_global"]).union(
        set(summaries["dev"]["thickness_values_global"])
    )))

    thk_values_by_mat = {}
    for m in global_materials:
        vals = set()
        vals.update(summaries["train"]["thickness_values_by_material"].get(m, []))
        vals.update(summaries["dev"]["thickness_values_by_material"].get(m, []))
        thk_values_by_mat[m] = sorted(list(vals))

    # Overview json
    overview = {
        "data_dir": DATA_DIR,
        "splits_used": splits,
        "plot_mode": PLOT_MODE,
        "max_curves_per_split": MAX_CURVES_PER_SPLIT,
        "num_materials_union": len(global_materials),
        "materials_union": global_materials,
        "num_head_pair_types": len(global_head_pairs),
        "num_adjacent_pair_types": len(global_adj_pairs),
        "thickness_values_global": global_thk_values,
    }

    # Save json
    out_json = os.path.join(OUT_DIR, "summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"overview": overview, "by_split": summaries}, f, ensure_ascii=False, indent=2)

    # Save txt (human-friendly)
    out_txt = os.path.join(OUT_DIR, "summary.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"DATA_DIR: {DATA_DIR}\n")
        f.write(f"SPLITS_USED: {splits}\n")
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
            f.write(f"global_thickness_values(nm): {s['thickness_values_global']}\n")
            f.write(f"head_pair_types: {len(s['head_pair_distribution'])}\n")
            f.write(f"adjacent_pair_types: {len(s['adjacent_pair_distribution'])}\n\n")

        f.write("== UNION (train+dev) ==\n")
        f.write(f"num_materials_union: {len(global_materials)}\n")
        f.write(f"materials_union: {global_materials}\n")
        f.write(f"num_head_pair_types_union: {len(global_head_pairs)}\n")
        f.write(f"num_adjacent_pair_types_union: {len(global_adj_pairs)}\n")
        f.write(f"thickness_values_global_union(nm): {global_thk_values}\n")

    # ============ REPORT BLOCK print ============
    report_block = make_report_block(
        train_sum=summaries["train"],
        dev_sum=summaries["dev"],
        global_materials=global_materials,
        global_head_pairs=global_head_pairs,
        global_adj_pairs=global_adj_pairs,
        global_thk_values=global_thk_values,
        thk_values_by_mat=thk_values_by_mat,
    )

    # Also save report block as a separate file for convenience
    out_report = os.path.join(OUT_DIR, "report_block.txt")
    with open(out_report, "w", encoding="utf-8") as f:
        f.write(report_block + "\n")

    print(report_block)
    print("\n[Saved report block] ->", out_report)

    # ============ Plot overlays ============
    curves_all = np.concatenate([curve_bank["train"], curve_bank["dev"]], axis=0)

    if PLOT_MODE.upper() == "R_T":
        dim = curves_all.shape[1]
        if dim % 2 != 0:
            raise ValueError("R_T mode expects even dim (R+T).")
        W = dim // 2
        wl = wavelengths_ref
        if wl is None or len(wl) != W:
            wl = np.linspace(LAMBDA0, LAMBDA1, W)

        out_r = os.path.join(OUT_DIR, "curves_overlay_R.png")
        out_t = os.path.join(OUT_DIR, "curves_overlay_T.png")
        plot_overlay(curves_all[:, :W], wl, f"Overlay curves (R) | sampled={curves_all.shape[0]} | train+dev", out_r)
        plot_overlay(curves_all[:, W:], wl, f"Overlay curves (T) | sampled={curves_all.shape[0]} | train+dev", out_t)
    else:
        wl = wavelengths_ref
        if wl is None:
            wl = np.linspace(LAMBDA0, LAMBDA1, curves_all.shape[1])
        out_png = os.path.join(OUT_DIR, f"curves_overlay_{PLOT_MODE}.png")
        plot_overlay(curves_all, wl, f"Overlay curves ({PLOT_MODE}) | sampled={curves_all.shape[0]} | train+dev", out_png)

    print("\nDone.")
    print("Saved:")
    print("  summary.json     ->", out_json)
    print("  summary.txt      ->", out_txt)
    print("  report_block.txt ->", out_report)
    print("  plots / samples  ->", OUT_DIR)


if __name__ == "__main__":
    main()
