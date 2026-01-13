#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_dbr_v3_pairdist.py

在你已有的 eval_dbr_v2_debug.py 基础上：
- 保留 Oracle MAE / Pred MAE / 结构合法性诊断
- 新增：统计 PRED 的 pair 分布（包括 INVALID）
- 仍保持 wavelength grid / nm round 与生成器一致

用法示例：
python eval_dbr_v3_pairdist.py \
  --ckpt saved_models/optogpt/dbr_60k/model_inverse_best.pt \
  --dev_struct ./dataset/dbr/Structure_dev.pkl \
  --dev_spec   ./dataset/dbr/Spectrum_dev.pkl \
  --nk_dir     ./dataset/data \
  --lambda0 0.9 --lambda1 1.7 --step_um 0.005 \
  --max_len 64 \
  --num_eval 200 \
  --print_first_k 5
"""

import os
import argparse
import random
import pickle as pkl
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from scipy.interpolate import interp1d
from tmm import coh_tmm

# ====== optogpt 依赖 ======
from core.models.transformer import make_model_I, subsequent_mask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# 1) nk 加载：从 csv(wl,n,k) 插值到 wavelengths_um
# -------------------------
def load_nk(nk_dir, materials, wavelengths_um):
    nk = {}
    for mat in materials:
        path = os.path.join(nk_dir, f"{mat}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing nk csv: {path}")

        df = pd.read_csv(path).dropna()
        wl = df["wl"].values
        n = df["n"].values
        k = df["k"].values

        n_interp = interp1d(wl, n, bounds_error=False, fill_value="extrapolate")
        k_interp = interp1d(wl, k, bounds_error=False, fill_value="extrapolate")
        nk[mat] = n_interp(wavelengths_um) + 1j * k_interp(wavelengths_um)
    return nk


# -------------------------
# 2) DBR 光谱计算：输出 [R..., T...] 维度 = 2*N
# -------------------------
def calc_RT(materials, thicknesses_nm, nk_dict, wavelengths_um, pol="s", theta_deg=0.0):
    R, T = [], []
    d_list = [np.inf] + list(thicknesses_nm) + [np.inf]
    th0 = np.deg2rad(theta_deg)

    wl_nm_list = np.round(wavelengths_um * 1000).astype(int)  # 与生成器一致
    for i, wl_nm in enumerate(wl_nm_list):
        n_list = [1] + [nk_dict[m][i] for m in materials] + [1]
        res = coh_tmm(pol=pol, n_list=n_list, d_list=d_list, th_0=th0, lam_vac=int(wl_nm))
        R.append(res["R"])
        T.append(res["T"])

    R = np.asarray(R, dtype=np.float32)
    T = np.asarray(T, dtype=np.float32)
    return np.concatenate([R, T], axis=0)  # (2*N,)


# -------------------------
# 3) token 解析："TiO2_158" -> (TiO2, 158)
# -------------------------
def parse_structure_tokens(tokens):
    mats, thks = [], []
    for s in tokens:
        if "_" not in s:
            continue
        m, t = s.split("_", 1)
        try:
            mats.append(m)
            thks.append(float(t))
        except Exception:
            continue
    return mats, thks


def infer_pair_name_from_tokens(tokens):
    """
    tokens: list[str], e.g. ["TiO2_158", "SiO2_212", ...]
    取前两层材料作为 pair_name
    """
    mats = []
    for s in tokens:
        if "_" not in s:
            continue
        mats.append(s.split("_", 1)[0])
        if len(mats) >= 2:
            break
    if len(mats) < 2:
        return "INVALID"
    return f"{mats[0]}/{mats[1]}"


# -------------------------
# 4) Greedy decode（保证 src shape = (1,1,spec_dim)）
# -------------------------
@torch.no_grad()
def greedy_decode(model, struc_index_dict, struc_word_dict, spec_target, max_len, start_symbol="BOS"):
    bos_id = struc_word_dict[start_symbol]
    ys = torch.ones(1, 1, dtype=torch.long, device=DEVICE).fill_(bos_id)

    spec_np = np.asarray(spec_target, dtype=np.float32)
    src = torch.from_numpy(spec_np).to(DEVICE)[None, None, :]  # (1,1,spec_dim)
    src_mask = None

    out_tokens = []
    for _ in range(max_len - 1):
        trg_mask = Variable(subsequent_mask(ys.size(1)).type_as(src.data)).to(DEVICE)
        out = model(src, Variable(ys), src_mask, trg_mask)

        prob = model.generator(out[:, -1])  # (1, vocab), log_softmax
        _, next_word = torch.max(prob, dim=1)
        next_id = int(next_word.item())

        ys = torch.cat(
            [ys, torch.ones(1, 1, dtype=torch.long, device=DEVICE).fill_(next_id)],
            dim=1
        )

        sym = struc_index_dict.get(next_id, "UNK")
        if sym == "EOS":
            break
        out_tokens.append(sym)

    return out_tokens


# -------------------------
# 5) 结构有效性检查 + 统计
# -------------------------
SPECIAL = {"UNK", "PAD", "BOS", "EOS"}

def check_structure(tokens, allowed_materials=None):
    """
    返回：ok(bool), info(dict)
    - ok=True 表示基本可用（至少能 parse 出 >=1 层，并且无明显特殊 token 污染）
    """
    info = {}
    info["len_tokens"] = len(tokens)
    info["num_special"] = sum(1 for t in tokens if t in SPECIAL)
    info["has_special"] = info["num_special"] > 0

    mats, thks = parse_structure_tokens(tokens)
    info["num_layers_parsed"] = len(mats)

    if len(mats) == 0:
        info["reason"] = "empty_parsed"
        return False, info

    # 是否交替（DBR 通常交替）
    if len(mats) >= 2:
        info["is_alternating"] = all(mats[i] != mats[i - 1] for i in range(1, len(mats)))
    else:
        info["is_alternating"] = True

    # 材料是否在允许集合中
    if allowed_materials is not None:
        ood = [m for m in mats if m not in allowed_materials]
        info["num_ood_materials"] = len(ood)
        info["ood_materials_sample"] = sorted(list(set(ood)))[:5]
    else:
        info["num_ood_materials"] = 0
        info["ood_materials_sample"] = []

    # 厚度基本范围（nm，宽松阈值）
    thks_np = np.asarray(thks, dtype=np.float32)
    info["thk_min"] = float(thks_np.min())
    info["thk_max"] = float(thks_np.max())
    info["thk_mean"] = float(thks_np.mean())

    if info["has_special"]:
        info["reason"] = "contains_special_token"
        return False, info
    if info["num_ood_materials"] > 0:
        info["reason"] = "ood_material"
        return False, info
    if info["thk_min"] <= 0 or info["thk_max"] > 5000:
        info["reason"] = "bad_thickness_range"
        return False, info

    info["reason"] = "ok"
    return True, info


def load_pickle(path):
    with open(path, "rb") as f:
        return pkl.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="model checkpoint .pt")
    ap.add_argument("--dev_struct", type=str, default="./dataset/dbr/Structure_dev.pkl")
    ap.add_argument("--dev_spec", type=str, default="./dataset/dbr/Spectrum_dev.pkl")
    ap.add_argument("--nk_dir", type=str, default="./dataset/data")

    # wavelength grid: 与生成器一致（0.005步长，round到nm）
    ap.add_argument("--lambda0", type=float, default=0.9)
    ap.add_argument("--lambda1", type=float, default=1.7)
    ap.add_argument("--step_um", type=float, default=0.005)

    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--num_eval", type=int, default=200, help="<=0 means all")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--print_first_k", type=int, default=5, help="print debug for first k samples")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- load ckpt ----
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["configs"]

    # ---- build model ----
    model = make_model_I(
        cfg.spec_dim,
        cfg.struc_dim,
        cfg.layers,
        cfg.d_model,
        cfg.d_ff,
        cfg.head_num,
        cfg.dropout,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---- dicts ----
    struc_word_dict = cfg.struc_word_dict
    struc_index_dict = cfg.struc_index_dict

    # ---- load dev data ----
    dev_struct = load_pickle(args.dev_struct)  # list[list[str]]
    dev_spec = load_pickle(args.dev_spec)      # list[list[float]]
    dev_spec = np.asarray(dev_spec, dtype=np.float32)

    N = len(dev_spec)
    spec_dim = dev_spec.shape[1]
    assert spec_dim == cfg.spec_dim, f"spec_dim mismatch: dev_spec={spec_dim}, cfg={cfg.spec_dim}"

    # ---- wavelength grid ----
    n_pts = int(round((args.lambda1 - args.lambda0) / args.step_um)) + 1
    wavelengths_um = np.linspace(args.lambda0, args.lambda1, n_pts)
    wl_nm_list = np.round(wavelengths_um * 1000).astype(int)

    if spec_dim != 2 * n_pts:
        print("[FATAL] spec_dim != 2 * n_pts")
        print("  spec_dim:", spec_dim)
        print("  n_pts:", n_pts, "=> 2*n_pts:", 2 * n_pts)
        print("  (请检查：生成器 wavelength 点数 vs eval 的 lambda0/lambda1/step 是否一致)")
        return

    # ---- materials from dev_struct ----
    mats_set = set()
    for seq in dev_struct:
        for s in seq:
            if "_" in s:
                mats_set.add(s.split("_", 1)[0])
    mats = sorted(list(mats_set))
    if len(mats) == 0:
        print("[FATAL] No materials parsed from dev_struct. Check dev_struct content.")
        return
    print("materials(from dev_struct):", mats)

    # ---- load nk ----
    nk_dict = load_nk(args.nk_dir, mats, wavelengths_um)

    # ---- choose subset ----
    idxs = list(range(N))
    if args.num_eval > 0 and args.num_eval < N:
        idxs = random.sample(idxs, args.num_eval)

    # ---- diagnostics containers ----
    mae_oracle = []
    mae_pred = []
    bad_pred = 0
    bad_oracle = 0

    stats = {
        "pred_ok": 0,
        "pred_has_special": 0,
        "pred_ood_material": 0,
        "pred_not_alternating": 0,
        "pred_empty": 0,
        "pred_bad_thk": 0,
    }

    # ---- NEW: pred pair distribution ----
    pred_pair_counter = Counter()

    print("dev_spec range: min=%.4f max=%.4f mean=%.4f" %
          (float(dev_spec.min()), float(dev_spec.max()), float(dev_spec.mean())))
    print("wl_nm_list head/tail:", wl_nm_list[:5].tolist(), "...", wl_nm_list[-5:].tolist())

    # ---- loop ----
    for j, ii in enumerate(idxs):
        spec_target = dev_spec[ii]  # (spec_dim,)
        gt_tokens = dev_struct[ii]  # list[str]

        # ========== (1) ORACLE ==========
        mats_gt, thks_gt = parse_structure_tokens(gt_tokens)
        if len(mats_gt) == 0:
            bad_oracle += 1
        else:
            try:
                spec_gt = calc_RT(mats_gt, thks_gt, nk_dict, wavelengths_um)
                mae0 = float(np.mean(np.abs(spec_gt - spec_target)))
                mae_oracle.append(mae0)
            except Exception:
                bad_oracle += 1

        # ========== (2) PRED ==========
        pred_tokens = greedy_decode(
            model=model,
            struc_index_dict=struc_index_dict,
            struc_word_dict=struc_word_dict,
            spec_target=spec_target,
            max_len=args.max_len,
            start_symbol="BOS",
        )

        # NEW: record predicted pair (even if invalid)
        pred_pair_counter[infer_pair_name_from_tokens(pred_tokens)] += 1

        ok, info = check_structure(pred_tokens, allowed_materials=set(mats))
        if not ok:
            bad_pred += 1
            if info.get("reason") == "empty_parsed":
                stats["pred_empty"] += 1
            elif info.get("reason") == "contains_special_token":
                stats["pred_has_special"] += 1
            elif info.get("reason") == "ood_material":
                stats["pred_ood_material"] += 1
            elif info.get("reason") == "bad_thickness_range":
                stats["pred_bad_thk"] += 1
        else:
            stats["pred_ok"] += 1
            if not info.get("is_alternating", True):
                stats["pred_not_alternating"] += 1

            mats_pred, thks_pred = parse_structure_tokens(pred_tokens)
            try:
                spec_pred = calc_RT(mats_pred, thks_pred, nk_dict, wavelengths_um)
                mae1 = float(np.mean(np.abs(spec_pred - spec_target)))
                mae_pred.append(mae1)
            except Exception:
                bad_pred += 1

        # ========== print debug ==========
        if j < args.print_first_k:
            print("\n---- sample", ii, "----")
            print("target[min/max/mean]:",
                  float(spec_target.min()), float(spec_target.max()), float(spec_target.mean()))
            print("GT tokens head:", gt_tokens[:10], "... len=", len(gt_tokens))
            if len(mats_gt) > 0:
                alt = all(mats_gt[k] != mats_gt[k - 1] for k in range(1, len(mats_gt))) if len(mats_gt) > 1 else True
                print("GT pair:", infer_pair_name_from_tokens(gt_tokens), "| layers:", len(mats_gt), "| alt:", alt)
            print("PRED tokens head:", pred_tokens[:10], "... len=", len(pred_tokens))
            print("PRED pair:", infer_pair_name_from_tokens(pred_tokens))
            print("PRED check:", ok, info)
            if len(mae_oracle) > 0:
                print("oracle MAE(last):", mae_oracle[-1])
            if len(mae_pred) > 0:
                print("pred   MAE(last):", mae_pred[-1])

    # ---- report ----
    mae_oracle = np.asarray(mae_oracle, dtype=np.float32)
    mae_pred = np.asarray(mae_pred, dtype=np.float32)

    print("\n==================== REPORT ====================")
    print(f"eval requested samples: {len(idxs)}")

    # Oracle
    print("\n[ORACLE] GT structure -> TMM vs dev_spec")
    print(f"  valid oracle samples: {len(mae_oracle)}")
    print(f"  oracle failed       : {bad_oracle}")
    if len(mae_oracle) > 0:
        print(f"  oracle MAE mean  : {mae_oracle.mean():.6f}")
        print(f"  oracle MAE median: {np.median(mae_oracle):.6f}")
        print(f"  oracle MAE p90   : {np.quantile(mae_oracle, 0.90):.6f}")
        print(f"  oracle MAE p99   : {np.quantile(mae_oracle, 0.99):.6f}")
        if float(mae_oracle.mean()) > 1e-2:
            print("  !! WARNING: oracle MAE 不接近 0，说明 dev_spec 与 dev_struct/TMM/波长网格/nk 之间存在不一致。")

    # Pred
    print("\n[PRED] Model decode -> TMM vs dev_spec")
    print(f"  valid pred samples  : {len(mae_pred)}")
    print(f"  pred failed/invalid : {bad_pred}")
    if len(mae_pred) > 0:
        print(f"  pred MAE mean  : {mae_pred.mean():.6f}")
        print(f"  pred MAE median: {np.median(mae_pred):.6f}")
        print(f"  pred MAE p90   : {np.quantile(mae_pred, 0.90):.6f}")
        print(f"  pred MAE p99   : {np.quantile(mae_pred, 0.99):.6f}")

    # Structure diag
    print("\n[STRUCT DIAG]")
    for k, v in stats.items():
        print(f"  {k:22s}: {v}")

    # NEW: Pred pair distribution
    print("\n[PRED pair distribution]")
    total_pred = sum(pred_pair_counter.values())
    for k, v in pred_pair_counter.most_common():
        print(f"  {k:20s} {v:6d} ({v / max(total_pred, 1):.3%})")

    print("\nHow to interpret:")
    print("1) 如果 oracle MAE ~ 0（比如 0.001~0.01），说明数据(dev_struct/dev_spec)与TMM计算一致，数据没问题。")
    print("2) 如果 oracle MAE 也很大（0.1~0.5），几乎必然是：wavelength grid / nm rounding / nk插值 / dev_spec 排列顺序(R/T) 不一致。")
    print("3) 如果 oracle MAE 很小但 pred MAE 很大，同时 pred pair 分布极度集中：模型/解码发生坍塌（与数据pair分布无关）。")
    print("================================================")


if __name__ == "__main__":
    main()

"""
用法示例：

python eval_dbr.py \
  --ckpt saved_models/optogpt/dbr/best.pt \
  --dev_struct ./dataset/Structure_dev.pkl \
  --dev_spec ./dataset/Spectrum_dev.pkl \
  --nk_dir ./dataset/data \
  --lambda0 0.9 --lambda1 1.7 --step_um 0.005 \
  --max_len 64 \
  --num_eval 200 \
  --print_first_k 5
"""
