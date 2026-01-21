#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
summarize_optogpt_dataset_v2_report.py

v2.1 (this full version adds configurable overlay plotting)
Changes vs your v2:
1) Add argparse so you can set:
   - --data_dir
   - --plot_mode {R,T,R_T}
   - --max_curves_per_split (how many curves to save/sample per split)
   - --overlay_max_curves (how many curves to overlay in final plot, after merge)
   - --random_seed
2) Overlay plotting now re-samples again to avoid too many curves.

Assumes:
- Structure_{train,dev}.pkl: List[List[str]] token="Material_ThicknessNm"
- Spectrum_{train,dev}.pkl : List[List[float]] each spec is [R..., T...] (2W)
- meta_{train,dev}.pkl optional
"""

import os
import json
import argparse
import pickle as pkl
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import matplotlib.pyplot as plt


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
        m, _ = parse_token(tok)
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
        m, _ = parse_token(tok)
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
    if k is None or k <= 0:
        return np.arange(n, dtype=np.int64)
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
def summarize_split(
    data_dir: str,
    out_dir: str,
    split: str,
    plot_mode: str,
    lambda0: float,
    lambda1: float,
    step_um: float,
    max_curves_per_split: int,
    random_seed: int,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    struct_path = os.path.join(data_dir, f"Structure_{split}.pkl")
    spec_path   = os.path.join(data_dir, f"Spectrum_{split}.pkl")
    meta_path   = os.path.join(data_dir, f"meta_{split}.pkl")

    if not (os.path.exists(struct_path) and os.path.exists(spec_path)):
        raise FileNotFoundError(f"Missing files for split={split} in {data_dir}")

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
    specs_s = np.stack([slice_spec(specs[i], plot_mode) for i in range(n)], axis=0)
    dim = specs_s.shape[1]

    # Wavelengths
    if plot_mode.upper() == "R_T":
        if dim % 2 != 0:
            raise ValueError("R_T mode expects even dim (R+T).")
        W = dim // 2
        wl = np.linspace(lambda0, lambda1, int(round((lambda1 - lambda0) / step_um)) + 1)
        if len(wl) != W:
            wl = np.linspace(lambda0, lambda1, W)
    else:
        wl = np.linspace(lambda0, lambda1, int(round((lambda1 - lambda0) / step_um)) + 1)
        if len(wl) != dim:
            wl = np.linspace(lambda0, lambda1, dim)

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

    # Sample curves for overlay & saving
    rng = np.random.default_rng(random_seed + (0 if split == "train" else 1))
    sample_ids = safe_sample_indices(n, max_curves_per_split, rng)
    curves_sample = specs_s[sample_ids]
    np.save(os.path.join(out_dir, f"curves_sample_{split}_{plot_mode}.npy"), curves_sample)

    out = {
        "split": split,
        "num_samples": int(n),
        "spec_dim_after_slice": int(dim),
        "plot_mode": plot_mode,
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
        "_curves_sample_path": os.path.join(out_dir, f"curves_sample_{split}_{plot_mode}.npy"),
    }
    return out, curves_sample, wl


def make_report_block(train_sum: Dict[str, Any], dev_sum: Dict[str, Any],
                      global_materials: List[str],
                      global_head_pairs: List[Tuple[str, int]],
                      global_adj_pairs: List[Tuple[str, int]],
                      global_thk_values: List[int],
                      thk_values_by_mat: Dict[str, List[int]],
                      report_max_list: int) -> str:
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

    # Pairs
    lines.append(f"2）组合种类（材料对）")
    lines.append(f"- Head pair（首两层材料对）种类数：{len(global_head_pairs)}")
    lines.append(f"- Adjacent pair（相邻层材料对）种类数：{len(global_adj_pairs)}")
    lines.append("")
    lines.append("Head pair 详细列表（pair: count）：")
    for k, v in global_head_pairs[:report_max_list]:
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("Adjacent pair 详细列表（pair: count）：")
    for k, v in global_adj_pairs[:report_max_list]:
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

    # Type distribution
    if train_sum.get("type_distribution") is not None or dev_sum.get("type_distribution") is not None:
        lines.append("5）类型分布（若 meta 存在）")
        lines.append(f"- Train: {train_sum.get('type_distribution')}")
        lines.append(f"- Dev  : {dev_sum.get('type_distribution')}")
        lines.append("")

    return "\n".join(lines)


# =========================
# Main
# =========================
def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./dataset/mix_optogpt_style_v2",
                   help="Dataset directory containing Structure_{train,dev}.pkl and Spectrum_{train,dev}.pkl")
    p.add_argument("--plot_mode", type=str, default="R_T", choices=["R", "T", "R_T"],
                   help="Plot mode: R_T plots R and T separately; R only; T only.")
    p.add_argument("--lambda0", type=float, default=0.9)
    p.add_argument("--lambda1", type=float, default=1.7)
    p.add_argument("--step_um", type=float, default=0.005)

    # sampling controls
    p.add_argument("--max_curves_per_split", type=int, default=800,
                   help="How many curves to sample (and save) per split: train/dev.")
    p.add_argument("--overlay_max_curves", type=int, default=200,
                   help="How many curves to overlay in final plot (after merging train+dev sampled curves). "
                        "Set <=0 to plot all merged sampled curves (not recommended).")
    p.add_argument("--random_seed", type=int, default=42)

    # report printing
    p.add_argument("--report_max_list", type=int, default=5000,
                   help="Max number of pair entries to print in report block (you said want detailed).")
    return p


def main():
    args = build_argparser().parse_args()

    data_dir = args.data_dir
    out_dir = os.path.join(data_dir, "_summary")
    os.makedirs(out_dir, exist_ok=True)

    splits = ["train", "dev"]

    summaries = {}
    curve_bank = {}
    wavelengths_ref = None

    for sp in splits:
        s, curves_sample, wl = summarize_split(
            data_dir=data_dir,
            out_dir=out_dir,
            split=sp,
            plot_mode=args.plot_mode,
            lambda0=args.lambda0,
            lambda1=args.lambda1,
            step_um=args.step_um,
            max_curves_per_split=args.max_curves_per_split,
            random_seed=args.random_seed,
        )
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
        "data_dir": data_dir,
        "splits_used": splits,
        "plot_mode": args.plot_mode,
        "max_curves_per_split": args.max_curves_per_split,
        "overlay_max_curves": args.overlay_max_curves,
        "num_materials_union": len(global_materials),
        "materials_union": global_materials,
        "num_head_pair_types": len(global_head_pairs),
        "num_adjacent_pair_types": len(global_adj_pairs),
        "thickness_values_global": global_thk_values,
    }

    # Save json
    out_json = os.path.join(out_dir, "summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"overview": overview, "by_split": summaries}, f, ensure_ascii=False, indent=2)

    # Save txt (human-friendly)
    out_txt = os.path.join(out_dir, "summary.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"DATA_DIR: {data_dir}\n")
        f.write(f"SPLITS_USED: {splits}\n")
        f.write(f"PLOT_MODE: {args.plot_mode}\n")
        f.write(f"MAX_CURVES_PER_SPLIT: {args.max_curves_per_split}\n")
        f.write(f"OVERLAY_MAX_CURVES: {args.overlay_max_curves}\n\n")

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
        report_max_list=args.report_max_list,
    )

    out_report = os.path.join(out_dir, "report_block.txt")
    with open(out_report, "w", encoding="utf-8") as f:
        f.write(report_block + "\n")

    print(report_block)
    print("\n[Saved report block] ->", out_report)

    # ============ Plot overlays ============
    # Note: curve_bank holds "sampled curves per split" already.
    curves_all = np.concatenate([curve_bank["train"], curve_bank["dev"]], axis=0)

    # Re-sample AGAIN for overlay to keep plot readable
    rng_overlay = np.random.default_rng(args.random_seed + 999)
    if args.overlay_max_curves is not None and args.overlay_max_curves > 0:
        ids = safe_sample_indices(curves_all.shape[0], args.overlay_max_curves, rng_overlay)
        curves_plot = curves_all[ids]
    else:
        curves_plot = curves_all

    if args.plot_mode.upper() == "R_T":
        dim = curves_plot.shape[1]
        if dim % 2 != 0:
            raise ValueError("R_T mode expects even dim (R+T).")
        W = dim // 2
        wl = wavelengths_ref
        if wl is None or len(wl) != W:
            wl = np.linspace(args.lambda0, args.lambda1, W)

        out_r = os.path.join(out_dir, "curves_overlay_R.png")
        out_t = os.path.join(out_dir, "curves_overlay_T.png")
        plot_overlay(curves_plot[:, :W], wl, f"Overlay curves (R) | plotted={curves_plot.shape[0]} | train+dev", out_r)
        plot_overlay(curves_plot[:, W:], wl, f"Overlay curves (T) | plotted={curves_plot.shape[0]} | train+dev", out_t)
    else:
        wl = wavelengths_ref
        if wl is None:
            wl = np.linspace(args.lambda0, args.lambda1, curves_plot.shape[1])
        out_png = os.path.join(out_dir, f"curves_overlay_{args.plot_mode}.png")
        plot_overlay(curves_plot, wl, f"Overlay curves ({args.plot_mode}) | plotted={curves_plot.shape[0]} | train+dev", out_png)

    print("\nDone.")
    print("Saved:")
    print("  summary.json     ->", out_json)
    print("  summary.txt      ->", out_txt)
    print("  report_block.txt ->", out_report)
    print("  plots / samples  ->", out_dir)
    print("\nTips:")
    print(f"  - Use --overlay_max_curves to control visibility (e.g., 50/100/200).")
    print(f"  - Use --max_curves_per_split to control how many curves get saved per split.")


if __name__ == "__main__":
    main()
