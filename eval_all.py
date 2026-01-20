#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_mix_dataset_report.py

为“大规模混合薄膜数据集（DBR/AR/FP/RANDOM）”做验证报告：
1) Oracle MAE: GT -> TMM_FAST vs dev_spec （检查数据是否自洽）
2) Pred MAE  : greedy decode -> TMM_FAST vs dev_spec
3) 分组统计：
   - by meta['type'] (DBR/AR/FP/RANDOM)
   - by meta['num_layers'] (长度桶)
4) 分布统计：
   - pred pair topK
   - pred tokens topK
5) 多解/歧义验证（spec-neighbors structure dispersion）：
   - 在 dev_spec 空间找 kNN
   - 统计邻居结构一致率 same_structure_rate
   - 统计邻居材料pattern一致率 same_material_pattern

Usage:
python eval_all.py \
  --ckpt saved_models/optogpt/all_new/best.pt \
  --dev_struct ./dataset/all_new/Structure_dev.pkl \
  --dev_spec   ./dataset/all_new/Spectrum_dev.pkl \
  --dev_meta   ./dataset/all_new/meta_dev.pkl \
  --nk_dir     ./dataset/data \
  --spec_type  R_T \
  --max_len 64 \
  --num_eval 2000 \
  --tmm_batch 128 \
  --nn_check_n 500 \
  --nn_k 5 \
  --print_k 8
"""

import os
import argparse
import random
import pickle as pkl
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from scipy.interpolate import interp1d

import tmm_fast
from core.models.transformer import make_model_I, subsequent_mask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMPLEX_DTYPE = torch.complex64
REAL_DTYPE = torch.float32


# -------------------------
# IO / seed
# -------------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pkl.load(f)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------
# token parsing helpers
# -------------------------
def parse_structure_tokens(tokens: List[str]) -> Tuple[List[str], List[float]]:
    mats, thks = [], []
    for s in tokens:
        if "_" not in s:
            continue
        m, t = s.split("_", 1)
        try:
            thk = float(t)
        except Exception:
            continue
        if thk < 0:
            continue
        mats.append(m)
        thks.append(thk)
    return mats, thks

def infer_pair_name(tokens: List[str]) -> str:
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

def material_pattern(tokens: List[str]) -> str:
    mats = []
    for s in tokens:
        if "_" in s:
            mats.append(s.split("_", 1)[0])
    if len(mats) == 0:
        return "INVALID"
    return "/".join(mats)

def collect_materials_from_struct(struct_list: List[List[str]]) -> List[str]:
    mats = set()
    for seq in struct_list:
        for s in seq:
            if "_" in s:
                mats.add(s.split("_", 1)[0])
    return sorted(list(mats))


# -------------------------
# spectrum slicing
# -------------------------
def slice_spec(spec: np.ndarray, spec_type: str) -> np.ndarray:
    spec_type = spec_type.upper()
    if spec_type == "R_T":
        return spec
    if spec.ndim != 1:
        raise ValueError("slice_spec expects 1D spec vector.")
    L = spec.shape[0]
    if L % 2 != 0:
        raise ValueError(f"spec length {L} not even, cannot split into R/T.")
    W = L // 2
    if spec_type == "R":
        return spec[:W]
    if spec_type == "T":
        return spec[W:]
    raise ValueError(f"Unknown spec_type: {spec_type}")


# -------------------------
# nk loader
# -------------------------
def load_nk_torch(nk_dir: str, materials: List[str], wavelengths_um: np.ndarray) -> Dict[str, torch.Tensor]:
    nk = {}
    for mat in materials:
        path = os.path.join(nk_dir, f"{mat}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing nk csv: {path}")

        df = pd.read_csv(path).dropna()
        wl = df["wl"].values.astype(np.float64)
        n = df["n"].values.astype(np.float64)
        k = df["k"].values.astype(np.float64)

        n_interp = interp1d(wl, n, bounds_error=False, fill_value="extrapolate")
        k_interp = interp1d(wl, k, bounds_error=False, fill_value="extrapolate")

        nk_np = n_interp(wavelengths_um) + 1j * k_interp(wavelengths_um)
        nk[mat] = torch.tensor(nk_np, dtype=COMPLEX_DTYPE, device=DEVICE)
    return nk


# -------------------------
# tmm_fast packing with exit medium
# -------------------------
def _pack_batch_to_tmm_fast(
    batch_mats: List[List[str]],
    batch_thks_nm: List[List[float]],
    nk_dict_torch: Dict[str, torch.Tensor],
    wl_m: torch.Tensor,
    exit_medium: Optional[str] = None,
    force_exit_k0: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = len(batch_mats)
    W = wl_m.shape[0]
    Lmax = max(len(x) for x in batch_mats) if B > 0 else 0

    d = torch.zeros((B, Lmax + 2), dtype=REAL_DTYPE, device=DEVICE)
    d[:, 0] = float("inf")
    d[:, -1] = float("inf")

    n = torch.empty((B, Lmax + 2, W), dtype=COMPLEX_DTYPE, device=DEVICE)
    n[:, 0, :] = (1.0 + 0.0j)

    if exit_medium is None:
        n[:, -1, :] = (1.0 + 0.0j)
    else:
        if exit_medium not in nk_dict_torch:
            raise KeyError(f"exit_medium={exit_medium} not in nk_dict_torch.")
        out_nk = nk_dict_torch[exit_medium]
        if force_exit_k0:
            out_nk = torch.real(out_nk).to(REAL_DTYPE).to(COMPLEX_DTYPE) + 0.0j
        n[:, -1, :] = out_nk

    for bi in range(B):
        mats = batch_mats[bi]
        thks = batch_thks_nm[bi]
        L = len(mats)

        if L > 0:
            d[bi, 1:1 + L] = torch.tensor(thks, dtype=REAL_DTYPE, device=DEVICE) * 1e-9
            for li, m in enumerate(mats, start=1):
                n[bi, li, :] = nk_dict_torch[m]

        if L < Lmax and L > 0:
            n_pad = n[bi, L, :].clone()
            n[bi, 1 + L:1 + Lmax, :] = n_pad

    return n, d


@torch.no_grad()
def calc_spec_tmmfast_batch(
    batch_mats: List[List[str]],
    batch_thks_nm: List[List[float]],
    nk_dict_torch: Dict[str, torch.Tensor],
    wl_m: torch.Tensor,
    theta_rad: torch.Tensor,
    pol: str = "s",
    exit_medium: Optional[str] = None,
    force_exit_k0: bool = True,
    spec_type: str = "R_T",
) -> np.ndarray:
    n, d = _pack_batch_to_tmm_fast(
        batch_mats, batch_thks_nm, nk_dict_torch, wl_m,
        exit_medium=exit_medium, force_exit_k0=force_exit_k0
    )
    out = tmm_fast.coh_tmm(pol, n, d, theta_rad, wl_m)

    R = out["R"]; T = out["T"]
    if R.ndim == 3:
        R = R[:, 0, :]; T = T[:, 0, :]
    elif R.ndim != 2:
        raise RuntimeError(f"Unexpected R shape: {tuple(R.shape)}")

    st = spec_type.upper()
    if st == "R_T":
        spec = torch.cat([R, T], dim=-1)
    elif st == "R":
        spec = R
    elif st == "T":
        spec = T
    else:
        raise ValueError(f"Unknown spec_type: {spec_type}")

    return spec.detach().cpu().float().numpy()


# -------------------------
# greedy decode
# -------------------------
@torch.no_grad()
def greedy_decode(model, struc_index_dict, struc_word_dict, spec_target, max_len, start_symbol="BOS"):
    bos_id = struc_word_dict[start_symbol]
    ys = torch.ones(1, 1, dtype=torch.long, device=DEVICE).fill_(bos_id)

    src = torch.from_numpy(np.asarray(spec_target, dtype=np.float32)).to(DEVICE)[None, None, :]
    src_mask = None

    out_tokens = []
    for _ in range(max_len - 1):
        trg_mask = Variable(subsequent_mask(ys.size(1)).type_as(src.data)).to(DEVICE)
        out = model(src, Variable(ys), src_mask, trg_mask)
        prob = model.generator(out[:, -1])

        next_id = int(prob.argmax(dim=1).item())
        ys = torch.cat([ys, torch.ones(1, 1, dtype=torch.long, device=DEVICE).fill_(next_id)], dim=1)

        sym = struc_index_dict.get(next_id, "UNK")
        if sym == "EOS":
            break
        out_tokens.append(sym)
    return out_tokens


def auto_exit_medium(mats_in_data: List[str], preferred: Optional[str]) -> Optional[str]:
    if preferred is None:
        return "Glass_Substrate" if "Glass_Substrate" in mats_in_data else None
    p = preferred.strip().lower()
    if p in ["air", "none", "null"]:
        return None
    return preferred.strip()


def mae_per_sample(pred: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(pred - tgt), axis=1)


def bucket_by_layers(n: int) -> str:
    if n <= 1: return "L=1"
    if n == 2: return "L=2"
    if n == 3: return "L=3"
    if 4 <= n <= 5: return "L=4-5"
    if 6 <= n <= 10: return "L=6-10"
    if 11 <= n <= 18: return "L=11-18"
    return "L>=19"


def knn_ambiguity_check(spec: np.ndarray, struct_tokens: List[List[str]], meta: List[dict], check_n: int, nn_k: int, seed: int):
    """
    spec-neighbors structure dispersion
    - 在 spec 空间找最近邻
    - 统计结构是否一致/材料pattern是否一致
    """
    rng = np.random.default_rng(seed)
    N = spec.shape[0]
    idxs = rng.choice(N, size=min(check_n, N), replace=False)

    # L2 normalize for cosine distance
    x = spec.astype(np.float32)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)

    same_struct = 0
    same_pattern = 0
    total_pairs = 0
    nn_dists = []

    for ii in idxs:
        v = x[ii:ii+1]  # [1,D]
        # cosine dist = 1 - dot
        dots = (x @ v.T).squeeze(1)  # [N]
        dist = 1.0 - dots
        dist[ii] = 1e9  # exclude itself
        nn = np.argpartition(dist, nn_k)[:nn_k]
        nn = nn[np.argsort(dist[nn])]

        gt_seq = struct_tokens[ii]
        gt_pat = material_pattern(gt_seq)

        for j in nn:
            nn_dists.append(float(dist[j]))
            total_pairs += 1
            if struct_tokens[j] == gt_seq:
                same_struct += 1
            if material_pattern(struct_tokens[j]) == gt_pat:
                same_pattern += 1

    return {
        "check_n": int(len(idxs)),
        "nn_k": int(nn_k),
        "mean_nn_dist": float(np.mean(nn_dists)) if nn_dists else 0.0,
        "same_structure_rate": float(same_struct / max(total_pairs, 1)),
        "same_material_pattern": float(same_pattern / max(total_pairs, 1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--dev_struct", type=str, required=True)
    ap.add_argument("--dev_spec", type=str, required=True)
    ap.add_argument("--dev_meta", type=str, required=True)

    ap.add_argument("--nk_dir", type=str, default="./dataset/data")
    ap.add_argument("--lambda0", type=float, default=0.9)
    ap.add_argument("--lambda1", type=float, default=1.7)
    ap.add_argument("--step_um", type=float, default=0.005)

    ap.add_argument("--spec_type", type=str, default="R_T")
    ap.add_argument("--exit_medium", type=str, default=None)  # air/none/material
    ap.add_argument("--force_exit_k0", action="store_true", help="force exit medium k=0 (recommended for substrate)")
    ap.add_argument("--pol", type=str, default="s")

    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--num_eval", type=int, default=2000, help="<=0 means all")
    ap.add_argument("--print_k", type=int, default=8)
    ap.add_argument("--tmm_batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--pair_topk", type=int, default=10)
    ap.add_argument("--token_topk", type=int, default=10)

    ap.add_argument("--nn_check_n", type=int, default=500)
    ap.add_argument("--nn_k", type=int, default=5)

    args = ap.parse_args()
    set_seed(args.seed)

    # load data
    dev_struct = load_pickle(args.dev_struct)
    dev_meta = load_pickle(args.dev_meta)
    dev_spec_all = np.asarray(load_pickle(args.dev_spec), dtype=np.float32)

    if not (len(dev_struct) == len(dev_meta) == len(dev_spec_all)):
        raise ValueError("dev_struct/dev_meta/dev_spec length mismatch")

    # wavelength grid
    n_pts = int(round((args.lambda1 - args.lambda0) / args.step_um)) + 1
    wavelengths_um = np.linspace(args.lambda0, args.lambda1, n_pts)
    wl_m = torch.tensor(wavelengths_um * 1e-6, dtype=REAL_DTYPE, device=DEVICE)
    theta_rad = torch.tensor([0.0], dtype=REAL_DTYPE, device=DEVICE)

    # slice spec
    dev_spec = np.stack([slice_spec(dev_spec_all[i], args.spec_type) for i in range(len(dev_spec_all))], axis=0)

    # load model
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["configs"]
    model = make_model_I(cfg.spec_dim, cfg.struc_dim, cfg.layers, cfg.d_model, cfg.d_ff, cfg.head_num, cfg.dropout).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if dev_spec.shape[1] != cfg.spec_dim:
        raise ValueError(f"spec_dim mismatch: dev_spec={dev_spec.shape[1]} vs ckpt={cfg.spec_dim}")

    struc_word_dict = cfg.struc_word_dict
    struc_index_dict = cfg.struc_index_dict

    # choose eval subset
    N = len(dev_spec)
    idxs = list(range(N))
    if args.num_eval > 0 and args.num_eval < N:
        idxs = random.sample(idxs, args.num_eval)

    # materials & exit medium
    mats_in_data = collect_materials_from_struct([dev_struct[i] for i in idxs])
    exit_medium = auto_exit_medium(mats_in_data, args.exit_medium)

    materials_to_load = set(mats_in_data)
    if exit_medium is not None:
        materials_to_load.add(exit_medium)
    nk_dict = load_nk_torch(args.nk_dir, sorted(list(materials_to_load)), wavelengths_um)

    # containers
    oracle_mats, oracle_thks, targets = [], [], []
    pred_mats, pred_thks = [], []
    pred_tokens_list = []
    pair_counter = Counter()
    token_counter = Counter()

    type_list = []
    layer_list = []

    # build lists
    for ii in idxs:
        spec_tgt = dev_spec[ii]
        gt_tokens = dev_struct[ii]
        meta = dev_meta[ii]

        # meta keys: type / num_layers
        typ = meta.get("type", "UNKNOWN")
        nl = int(meta.get("num_layers", len(gt_tokens)))

        type_list.append(typ)
        layer_list.append(nl)

        mats_gt, thks_gt = parse_structure_tokens(gt_tokens)
        oracle_mats.append(mats_gt)
        oracle_thks.append(thks_gt)
        targets.append(spec_tgt)

        pred_tokens = greedy_decode(model, struc_index_dict, struc_word_dict, spec_tgt, args.max_len)
        pred_tokens_list.append(pred_tokens)

        pair_counter[infer_pair_name(pred_tokens)] += 1
        for t in pred_tokens:
            token_counter[t] += 1

        mats_pd, thks_pd = parse_structure_tokens(pred_tokens)
        pred_mats.append(mats_pd)
        pred_thks.append(thks_pd)

    # ensure nk coverage for predicted materials (lazy-load)
    def ensure_nk(batch_pred_mats: List[List[str]]):
        new_mats = set()
        for mats in batch_pred_mats:
            for m in mats:
                if m not in nk_dict:
                    new_mats.add(m)
        if new_mats:
            nk_dict.update(load_nk_torch(args.nk_dir, sorted(list(new_mats)), wavelengths_um))

    # batch compute oracle/pred mae
    B = max(1, int(args.tmm_batch))
    oracle_mae = []
    pred_mae = []

    for st in range(0, len(idxs), B):
        ed = min(len(idxs), st + B)

        spec_or = calc_spec_tmmfast_batch(
            oracle_mats[st:ed], oracle_thks[st:ed], nk_dict,
            wl_m, theta_rad, pol=args.pol,
            exit_medium=exit_medium, force_exit_k0=args.force_exit_k0,
            spec_type=args.spec_type
        )
        tgt = np.asarray(targets[st:ed], dtype=np.float32)
        oracle_mae.extend(mae_per_sample(spec_or, tgt).tolist())

        ensure_nk(pred_mats[st:ed])
        spec_pd = calc_spec_tmmfast_batch(
            pred_mats[st:ed], pred_thks[st:ed], nk_dict,
            wl_m, theta_rad, pol=args.pol,
            exit_medium=exit_medium, force_exit_k0=args.force_exit_k0,
            spec_type=args.spec_type
        )
        pred_mae.extend(mae_per_sample(spec_pd, tgt).tolist())

    oracle_mae = np.asarray(oracle_mae, dtype=np.float32)
    pred_mae = np.asarray(pred_mae, dtype=np.float32)

    # print some samples
    for j in range(min(args.print_k, len(idxs))):
        ii = idxs[j]
        gt_tokens = dev_struct[ii]
        pred_tokens = pred_tokens_list[j]
        meta = dev_meta[ii]
        print(f"\n---- sample {ii} ----")
        print(f"type={meta.get('type','?')} | num_layers={meta.get('num_layers',len(gt_tokens))}")
        print(f"GT   pair: {infer_pair_name(gt_tokens)} | len={len(gt_tokens)}")
        print(f"PRED pair: {infer_pair_name(pred_tokens)} | len={len(pred_tokens)}")
        print("GT   head:", gt_tokens[:10], "..." if len(gt_tokens) > 10 else "")
        print("PRED head:", pred_tokens[:10], "..." if len(pred_tokens) > 10 else "")
        print(f"Oracle MAE: {oracle_mae[j]:.6f}")
        print(f"Pred   MAE: {pred_mae[j]:.6f}")

    # group stats by type
    by_type = defaultdict(list)
    by_layerbucket = defaultdict(list)

    for m, t, nl in zip(pred_mae.tolist(), type_list, layer_list):
        by_type[t].append(m)
        by_layerbucket[bucket_by_layers(nl)].append(m)

    print("\n==================== OVERALL ====================")
    print(f"eval samples: {len(idxs)} / total_dev={N}")
    print(f"spec_type={args.spec_type} | pol={args.pol} | exit_medium={exit_medium or 'air'} | force_exit_k0={args.force_exit_k0}")
    print(f"[Oracle MAE] mean={oracle_mae.mean():.6f} median={np.median(oracle_mae):.6f} p90={np.quantile(oracle_mae,0.90):.6f}")
    print(f"[Pred   MAE] mean={pred_mae.mean():.6f} median={np.median(pred_mae):.6f} p90={np.quantile(pred_mae,0.90):.6f}")

    print("\n================= BY TYPE =================")
    for k in sorted(by_type.keys()):
        arr = np.asarray(by_type[k], dtype=np.float32)
        print(f"{k:8s}  n={len(arr):6d}  mean={arr.mean():.6f}  median={np.median(arr):.6f}  p90={np.quantile(arr,0.90):.6f}")

    print("\n============= BY NUM_LAYERS BUCKET =============")
    order = ["L=1","L=2","L=3","L=4-5","L=6-10","L=11-18","L>=19"]
    for k in order:
        if k not in by_layerbucket: 
            continue
        arr = np.asarray(by_layerbucket[k], dtype=np.float32)
        print(f"{k:6s}  n={len(arr):6d}  mean={arr.mean():.6f}  median={np.median(arr):.6f}  p90={np.quantile(arr,0.90):.6f}")

    print("\n============= PRED DISTRIBUTIONS =============")
    total_pairs = sum(pair_counter.values())
    print(f"[Pred pair top {args.pair_topk}]")
    for k, v in pair_counter.most_common(args.pair_topk):
        print(f"  {k:20s} {v:8d} ({v/max(total_pairs,1):.3%})")

    total_tok = sum(token_counter.values())
    print(f"\n[Pred token top {args.token_topk}]")
    for k, v in token_counter.most_common(args.token_topk):
        print(f"  {k:20s} {v:8d} ({v/max(total_tok,1):.3%})")

    print("\n============= AMBIGUITY CHECK (spec-neighbors) =============")
    # 在同一个 idxs 子集上做 kNN（更公平）
    spec_sub = dev_spec[idxs]
    struct_sub = [dev_struct[i] for i in idxs]
    meta_sub = [dev_meta[i] for i in idxs]
    amb = knn_ambiguity_check(spec_sub, struct_sub, meta_sub, args.nn_check_n, args.nn_k, args.seed)
    print(f"[Ambiguity NN Check] (spec-neighbors structure dispersion)")
    print(f"  check_n={amb['check_n']} | nn_k={amb['nn_k']}")
    print(f"  mean_nn_dist           = {amb['mean_nn_dist']:.6f}")
    print(f"  same_structure_rate    = {amb['same_structure_rate']*100:.3f}%")
    print(f"  same_material_pattern  = {amb['same_material_pattern']*100:.3f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
