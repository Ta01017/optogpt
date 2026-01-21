#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
summarize_for_report.py

Report-friendly summary for OptoGPT-style dataset.
- Only uses TRAIN + DEV (ignores TEST).
- Prints copy-paste summary text:
  materials count/list
  pair count/topK (adjacent pairs + head pairs)
  thickness values (unique sorted) & per-material thickness ranges

Assumes:
- Structure_train.pkl / Structure_dev.pkl exist under DATA_DIR
- meta_train.pkl / meta_dev.pkl optional (if exists, prints type ratio)
"""

import os
import pickle as pkl
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any
import numpy as np

# ======================
# Config (change this)
# ======================
DATA_DIR = "./dataset/mix_optogpt_style_v2p1_noFP_noShortAR"  # <-- change
TOPK = 20                 # top-k pairs to print
SHOW_MATERIAL_LIST = True # print material names list
SHOW_THK_VALUES = True    # print unique thickness values set
MAX_THK_VALUES_PRINT = 200  # avoid printing thousands if too many

# Choose which splits to include
USE_TRAIN = True
USE_DEV = True

# ======================
# Helpers
# ======================
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
        thk = None
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

def safe_meta_type(m):
    if isinstance(m, dict) and "type" in m:
        return m["type"]
    return None

# ======================
# Main
# ======================
def main():
    splits = []
    if USE_TRAIN:
        splits.append("train")
    if USE_DEV:
        splits.append("dev")

    all_structs = []
    all_metas = []

    for sp in splits:
        sp_struct = os.path.join(DATA_DIR, f"Structure_{sp}.pkl")
        if not os.path.exists(sp_struct):
            raise FileNotFoundError(f"Missing: {sp_struct}")
        structs = load_pkl(sp_struct)
        all_structs.extend(structs)

        sp_meta = os.path.join(DATA_DIR, f"meta_{sp}.pkl")
        if os.path.exists(sp_meta):
            metas = load_pkl(sp_meta)
            if len(metas) == len(structs):
                all_metas.extend(metas)
            else:
                # keep alignment simple: if mismatch, skip meta
                all_metas = []

    N = len(all_structs)

    # stats containers
    materials = set()
    thk_values = set()
    mat2thk = defaultdict(list)

    head_pairs_cnt = Counter()
    adj_pairs_cnt = Counter()
    type_cnt = Counter()
    lens = []

    for i, toks in enumerate(all_structs):
        lens.append(len(toks))

        # meta types (optional)
        if all_metas:
            t = safe_meta_type(all_metas[i])
            if t is not None:
                type_cnt[t] += 1

        head_pairs_cnt[head_pair(toks)] += 1
        for p in adjacent_pairs(toks):
            adj_pairs_cnt[p] += 1

        for tok in toks:
            m, th = parse_token(tok)
            if m is None:
                continue
            materials.add(m)
            if th is not None:
                thk_values.add(th)
                mat2thk[m].append(th)

    # material thickness ranges
    mat_ranges = {}
    for m, arr in mat2thk.items():
        a = np.asarray(arr, dtype=np.int32)
        mat_ranges[m] = {
            "min": int(a.min()),
            "max": int(a.max()),
            "unique_count": int(len(set(a.tolist()))),
        }

    lens_np = np.asarray(lens, dtype=np.int32)
    len_stats = {
        "min": int(lens_np.min()),
        "mean": float(lens_np.mean()),
        "max": int(lens_np.max()),
        "p50": float(np.percentile(lens_np, 50)),
        "p90": float(np.percentile(lens_np, 90)),
        "p95": float(np.percentile(lens_np, 95)),
    }

    # Prepare report text
    mats_sorted = sorted(materials)
    thk_sorted = sorted(thk_values)
    num_pairs_adj = len(adj_pairs_cnt)
    num_pairs_head = len(head_pairs_cnt)

    # type ratio text
    type_text = ""
    if type_cnt:
        total_t = sum(type_cnt.values())
        parts = [f"{k}:{v}({v/total_t:.1%})" for k, v in type_cnt.most_common()]
        type_text = "；".join(parts)

    # Print (copy-paste friendly)
    print("\n==================== REPORT SUMMARY (TRAIN+DEV) ====================")
    print(f"数据规模（train+dev）: {N}")
    if type_text:
        print(f"结构类型分布: {type_text}")

    print(f"\n材料种类（num_materials）: {len(mats_sorted)}")
    if SHOW_MATERIAL_LIST:
        print("材料列表: " + ", ".join(mats_sorted))

    print(f"\n厚度离散取值（unique thickness values）: {len(thk_sorted)} 种")
    if thk_sorted:
        print(f"厚度范围: [{thk_sorted[0]}, {thk_sorted[-1]}] nm")
    if SHOW_THK_VALUES:
        if len(thk_sorted) <= MAX_THK_VALUES_PRINT:
            print("厚度取值集合(nm): " + ", ".join(map(str, thk_sorted)))
        else:
            head = thk_sorted[:MAX_THK_VALUES_PRINT//2]
            tail = thk_sorted[-MAX_THK_VALUES_PRINT//2:]
            print("厚度取值集合(nm): " + ", ".join(map(str, head)) + " ... " + ", ".join(map(str, tail)))
            print(f"(太多了，只展示前后各 {MAX_THK_VALUES_PRINT//2} 个)")

    print("\n每种材料厚度范围（min~max, unique_count）:")
    for m in mats_sorted:
        if m in mat_ranges:
            r = mat_ranges[m]
            print(f"  - {m:10s}: {r['min']:4d} ~ {r['max']:4d} nm  | unique_thk={r['unique_count']}")
        else:
            print(f"  - {m:10s}: (no thickness parsed)")

    print(f"\n组合种类（材料对）统计：")
    print(f"  - 相邻层材料对（adjacent pairs）种类数: {num_pairs_adj}")
    print(f"  - 首两层材料对（head pairs）种类数: {num_pairs_head}")

    print(f"\nTop-{TOPK} 相邻层材料对（adjacent pairs）:")
    total_adj = sum(adj_pairs_cnt.values()) if adj_pairs_cnt else 1
    for k, v in adj_pairs_cnt.most_common(TOPK):
        print(f"  {k:18s} {v:8d} ({v/total_adj:.1%})")

    print(f"\nTop-{TOPK} 首两层材料对（head pairs）:")
    total_head = sum(head_pairs_cnt.values()) if head_pairs_cnt else 1
    for k, v in head_pairs_cnt.most_common(TOPK):
        print(f"  {k:18s} {v:8d} ({v/total_head:.1%})")

    print("\n序列长度统计（#layers）:")
    print(f"  min={len_stats['min']}  mean={len_stats['mean']:.3f}  max={len_stats['max']}")
    print(f"  p50={len_stats['p50']:.1f}  p90={len_stats['p90']:.1f}  p95={len_stats['p95']:.1f}")

    # Also save a small txt you can attach to report
    out_txt = os.path.join(DATA_DIR, "_report_summary_train_dev.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        # dump same content
        import sys
        # quick trick: regenerate via prints to file (simple + robust)
        # We'll just write key sections
        f.write(f"数据规模（train+dev）: {N}\n")
        if type_text:
            f.write(f"结构类型分布: {type_text}\n")
        f.write(f"\n材料种类: {len(mats_sorted)}\n")
        f.write("材料列表: " + ", ".join(mats_sorted) + "\n")
        f.write(f"\n厚度离散取值: {len(thk_sorted)} 种\n")
        if thk_sorted:
            f.write(f"厚度范围: [{thk_sorted[0]}, {thk_sorted[-1]}] nm\n")
        if len(thk_sorted) <= 500:
            f.write("厚度取值集合(nm): " + ", ".join(map(str, thk_sorted)) + "\n")
        else:
            f.write("厚度取值集合(nm): (too many, see console)\n")

        f.write("\n每种材料厚度范围:\n")
        for m in mats_sorted:
            if m in mat_ranges:
                r = mat_ranges[m]
                f.write(f"  - {m}: {r['min']}~{r['max']} nm | unique_thk={r['unique_count']}\n")

        f.write(f"\n相邻层材料对种类数: {num_pairs_adj}\n")
        f.write(f"首两层材料对种类数: {num_pairs_head}\n")

        f.write(f"\nTop-{TOPK} adjacent pairs:\n")
        for k, v in adj_pairs_cnt.most_common(TOPK):
            f.write(f"  {k} {v}\n")

        f.write(f"\nTop-{TOPK} head pairs:\n")
        for k, v in head_pairs_cnt.most_common(TOPK):
            f.write(f"  {k} {v}\n")

        f.write("\n序列长度统计:\n")
        f.write(f"  min={len_stats['min']} mean={len_stats['mean']:.3f} max={len_stats['max']}\n")
        f.write(f"  p50={len_stats['p50']:.1f} p90={len_stats['p90']:.1f} p95={len_stats['p95']:.1f}\n")

    print(f"\n[Saved] {out_txt}")
    print("=====================================================================\n")


if __name__ == "__main__":
    main()
