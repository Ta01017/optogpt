# import pickle as pkl
# from collections import Counter

# def load(path):
#     with open(path, "rb") as f:
#         return pkl.load(f)

# def infer_pair_name(struct_seq):
#     """
#     struct_seq: list[str], e.g.
#     ["TiO2_158", "SiO2_212", "TiO2_160", ...]
#     """
#     mats = []
#     for s in struct_seq:
#         if "_" not in s:
#             continue
#         mats.append(s.split("_", 1)[0])
#         if len(mats) >= 2:
#             break
#     if len(mats) < 2:
#         return "INVALID"
#     return f"{mats[0]}/{mats[1]}"

# def count_pairs(struct_list):
#     c = Counter(infer_pair_name(seq) for seq in struct_list)
#     total = sum(c.values())
#     return c, total

# train_struct = load("./dataset/dbr/Structure_train.pkl")
# dev_struct   = load("./dataset/dbr/Structure_dev.pkl")

# for name, data in [("train", train_struct), ("dev", dev_struct)]:
#     c, total = count_pairs(data)
#     print(f"\n== {name} pair distribution ==")
#     for k, v in c.most_common():
#         print(f"{k:20s} {v:8d} ({v/total:.3%})")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataset_stats_dbr.py

统计 DBR 数据集结构 token（Material_Thickness）：
- pair_name 分布
- 材料频率
- 厚度分布（全局 + 分材料）
- 层数/对数分布
- 交替性（是否相邻材料相同）
- 可导出 csv

用法示例：
python dataset_stats_dbr.py \
  --struct_train ./dataset/dbr/Structure_train.pkl \
  --struct_dev   ./dataset/dbr/Structure_dev.pkl \
  --out_dir ./stats_out
"""

import os
import argparse
import pickle as pkl
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


SPECIAL = {"PAD", "BOS", "EOS", "UNK"}

def load_pkl(path):
    with open(path, "rb") as f:
        return pkl.load(f)

def parse_token(tok: str):
    """
    'TiO2_158' -> ('TiO2', 158)
    return (None, None) if cannot parse
    """
    if "_" not in tok:
        return None, None
    m, t = tok.split("_", 1)
    m = m.strip()
    if m in SPECIAL:
        return None, None
    try:
        d = int(float(t))
    except Exception:
        return None, None
    return m, d

def seq_to_mats_thks(seq):
    mats, thks = [], []
    for tok in seq:
        m, d = parse_token(tok)
        if m is None:
            continue
        mats.append(m)
        thks.append(d)
    return mats, thks

def infer_pair_name(seq):
    mats, _ = seq_to_mats_thks(seq)
    if len(mats) < 2:
        return "INVALID"
    return f"{mats[0]}/{mats[1]}"

def is_alternating(mats):
    if len(mats) <= 1:
        return True
    return all(mats[i] != mats[i-1] for i in range(1, len(mats)))

def describe_array(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return dict(count=0)
    return dict(
        count=int(arr.size),
        min=float(arr.min()),
        p01=float(np.quantile(arr, 0.01)),
        p05=float(np.quantile(arr, 0.05)),
        p10=float(np.quantile(arr, 0.10)),
        p25=float(np.quantile(arr, 0.25)),
        median=float(np.quantile(arr, 0.50)),
        p75=float(np.quantile(arr, 0.75)),
        p90=float(np.quantile(arr, 0.90)),
        p95=float(np.quantile(arr, 0.95)),
        p99=float(np.quantile(arr, 0.99)),
        max=float(arr.max()),
        mean=float(arr.mean()),
        std=float(arr.std()),
        unique=int(len(np.unique(arr)))
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--struct_train", type=str, required=True)
    ap.add_argument("--struct_dev", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./stats_out")
    ap.add_argument("--export_csv", action="store_true", help="export csv reports")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_struct = load_pkl(args.struct_train)
    dev_struct = load_pkl(args.struct_dev)

    def analyze(split_name, data):
        print(f"\n==================== {split_name} ====================")
        n = len(data)
        print("num samples:", n)

        pair_cnt = Counter()
        mat_cnt = Counter()
        thk_all = []
        layers_all = []
        not_alt = 0
        invalid_seq = 0

        mat2thks = defaultdict(list)
        mat2unique = defaultdict(set)

        # layer tokens frequency (full tokens)
        token_cnt = Counter()

        for seq in data:
            # raw token frequency（用于看是否被某些 token 统治）
            for tok in seq:
                token_cnt[tok] += 1

            mats, thks = seq_to_mats_thks(seq)
            if len(mats) == 0:
                invalid_seq += 1
                continue

            layers_all.append(len(mats))
            if not is_alternating(mats):
                not_alt += 1

            pair_cnt[infer_pair_name(seq)] += 1

            for m, d in zip(mats, thks):
                mat_cnt[m] += 1
                thk_all.append(d)
                mat2thks[m].append(d)
                mat2unique[m].add(d)

        # ---- basic ----
        print("invalid/empty parsed:", invalid_seq, f"({invalid_seq/max(n,1):.2%})")
        print("not alternating:", not_alt, f"({not_alt/max(n,1):.2%})")

        # ---- layers ----
        if layers_all:
            layers_desc = describe_array(layers_all)
            print("\n[num_layers stats]")
            for k in ["count","min","p10","median","p90","max","mean","std","unique"]:
                print(f"  {k:>6s}: {layers_desc[k]}")
        else:
            print("\n[num_layers stats] empty")

        # ---- pair distribution ----
        print("\n[pair_name distribution] (top 20)")
        total_pairs = sum(pair_cnt.values())
        for k, v in pair_cnt.most_common(20):
            print(f"  {k:20s} {v:8d} ({v/max(total_pairs,1):.3%})")
        # 也保存完整分布
        pair_df = pd.DataFrame(
            [(k, v, v/max(total_pairs,1)) for k, v in pair_cnt.items()],
            columns=["pair_name", "count", "ratio"]
        ).sort_values("count", ascending=False)

        # ---- material frequency ----
        print("\n[material frequency] (by layer tokens)")
        total_mat = sum(mat_cnt.values())
        mat_df = pd.DataFrame(
            [(k, v, v/max(total_mat,1), len(mat2unique[k])) for k, v in mat_cnt.items()],
            columns=["material", "count", "ratio", "unique_thickness"]
        ).sort_values("count", ascending=False)
        for _, r in mat_df.iterrows():
            print(f"  {r['material']:10s} cnt={int(r['count']):8d} "
                  f"ratio={float(r['ratio']):.3%} unique_thk={int(r['unique_thickness'])}")

        # ---- thickness global ----
        print("\n[thickness global stats] (nm)")
        thk_desc = describe_array(thk_all)
        for k in ["count","min","p01","p05","p10","median","p90","p95","p99","max","mean","std","unique"]:
            print(f"  {k:>6s}: {thk_desc[k]}")

        # ---- thickness per material ----
        print("\n[thickness per material stats] (nm)")
        rows = []
        for m in sorted(mat2thks.keys()):
            desc = describe_array(mat2thks[m])
            rows.append({
                "material": m,
                **desc,
            })
        thk_mat_df = pd.DataFrame(rows).sort_values("count", ascending=False)
        # 控制台打印 top
        print(thk_mat_df[["material","count","min","p10","median","p90","max","mean","std","unique"]].to_string(index=False))

        # ---- token frequency (top) ----
        print("\n[token frequency] (top 20 raw tokens, include specials)")
        total_tok = sum(token_cnt.values())
        for tok, c in token_cnt.most_common(20):
            print(f"  {tok:20s} {c:8d} ({c/max(total_tok,1):.3%})")

        # ---- export ----
        if args.export_csv:
            pair_df.to_csv(os.path.join(args.out_dir, f"{split_name}_pair_dist.csv"), index=False)
            mat_df.to_csv(os.path.join(args.out_dir, f"{split_name}_material_freq.csv"), index=False)
            thk_mat_df.to_csv(os.path.join(args.out_dir, f"{split_name}_thickness_by_material.csv"), index=False)
            # layers histogram-ish
            if layers_all:
                pd.Series(layers_all).value_counts().sort_index().to_csv(
                    os.path.join(args.out_dir, f"{split_name}_num_layers_counts.csv"),
                    header=["count"]
                )
            # global thk hist (bin=10nm)
            if thk_all:
                bins = (np.asarray(thk_all) // 10) * 10
                pd.Series(bins).value_counts().sort_index().to_csv(
                    os.path.join(args.out_dir, f"{split_name}_thickness_hist_10nm.csv"),
                    header=["count"]
                )

        return {
            "pair_df": pair_df,
            "mat_df": mat_df,
            "thk_mat_df": thk_mat_df,
            "token_cnt": token_cnt,
            "layers_all": layers_all,
            "thk_all": thk_all,
        }

    train_stats = analyze("train", train_struct)
    dev_stats = analyze("dev", dev_struct)

    # ---- compare train vs dev pair distribution overlap ----
    print("\n==================== train vs dev compare ====================")
    train_pairs = set(train_stats["pair_df"]["pair_name"].tolist())
    dev_pairs = set(dev_stats["pair_df"]["pair_name"].tolist())
    print("pair types train:", len(train_pairs), "| dev:", len(dev_pairs),
          "| intersection:", len(train_pairs & dev_pairs))

    train_mats = set(train_stats["mat_df"]["material"].tolist())
    dev_mats = set(dev_stats["mat_df"]["material"].tolist())
    print("materials train:", len(train_mats), "| dev:", len(dev_mats),
          "| intersection:", len(train_mats & dev_mats))

    if args.export_csv:
        print(f"\n[Saved CSV reports to] {args.out_dir}")

if __name__ == "__main__":
    main()
