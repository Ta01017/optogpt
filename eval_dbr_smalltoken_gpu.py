#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_dbr_simple_mae_tmmfast.py

最简 eval（tmm_fast 版，支持 batch）：
- Oracle MAE: GT -> TMM_FAST -> spec vs dev_spec
- Pred   MAE: greedy decode -> TMM_FAST -> spec vs dev_spec
- 打印若干样例：gt/pred pair + head + oracle/pred mae
- 输出整体 mae + pred pair 分布
"""

import os
import argparse
import random
import pickle as pkl
from collections import Counter
from typing import Dict, List, Tuple

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


def load_pickle(path):
    with open(path, "rb") as f:
        return pkl.load(f)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_structure_tokens(tokens):
    mats, thks = [], []
    for s in tokens:
        if "_" not in s:
            continue
        m, t = s.split("_", 1)
        try:
            mats.append(m)
            thks.append(float(t))  # nm
        except Exception:
            continue
    return mats, thks

def infer_pair_name(tokens):
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

def load_nk_torch(nk_dir, materials, wavelengths_um) -> Dict[str, torch.Tensor]:
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


def _pack_batch_to_tmm_fast(
    batch_mats: List[List[str]],
    batch_thks_nm: List[List[float]],
    nk_dict_torch: Dict[str, torch.Tensor],
    wl_m: torch.Tensor,  # [W]
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = len(batch_mats)
    W = wl_m.shape[0]
    Lmax = max(len(x) for x in batch_mats) if B > 0 else 0

    d = torch.zeros((B, Lmax + 2), dtype=REAL_DTYPE, device=DEVICE)
    d[:, 0] = float("inf")
    d[:, -1] = float("inf")

    n = torch.empty((B, Lmax + 2, W), dtype=COMPLEX_DTYPE, device=DEVICE)
    n[:, 0, :] = (1.0 + 0.0j)
    n[:, -1, :] = (1.0 + 0.0j)

    for bi in range(B):
        mats = batch_mats[bi]
        thks = batch_thks_nm[bi]
        L = len(mats)

        if L > 0:
            d[bi, 1:1 + L] = torch.tensor(thks, dtype=REAL_DTYPE, device=DEVICE) * 1e-9  # nm->m
            for li, m in enumerate(mats, start=1):
                n[bi, li, :] = nk_dict_torch[m]

        # pad: thickness=0, n copy last real layer (zero thickness won't affect)
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
) -> np.ndarray:
    n, d = _pack_batch_to_tmm_fast(batch_mats, batch_thks_nm, nk_dict_torch, wl_m)
    out = tmm_fast.coh_tmm(pol, n, d, theta_rad, wl_m)

    R = out["R"]
    T = out["T"]

    if R.ndim == 3:
        R = R[:, 0, :]
        T = T[:, 0, :]
    elif R.ndim != 2:
        raise RuntimeError(f"Unexpected R shape: {tuple(R.shape)}")

    spec = torch.cat([R, T], dim=-1)  # [B, 2W]
    return spec.detach().cpu().float().numpy()


@torch.no_grad()
def greedy_decode(model, struc_index_dict, struc_word_dict, spec_target, max_len, start_symbol="BOS"):
    bos_id = struc_word_dict[start_symbol]
    ys = torch.ones(1, 1, dtype=torch.long, device=DEVICE).fill_(bos_id)

    spec_np = np.asarray(spec_target, dtype=np.float32)
    src = torch.from_numpy(spec_np).to(DEVICE)[None, None, :]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--dev_struct", type=str, required=True)
    ap.add_argument("--dev_spec", type=str, required=True)
    ap.add_argument("--nk_dir", type=str, default="./dataset/data")
    ap.add_argument("--lambda0", type=float, default=0.9)
    ap.add_argument("--lambda1", type=float, default=1.7)
    ap.add_argument("--step_um", type=float, default=0.005)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--num_eval", type=int, default=200, help="<=0 means all")
    ap.add_argument("--print_k", type=int, default=5, help="print first k samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pair_topk", type=int, default=10)
    ap.add_argument("--tmm_batch", type=int, default=128, help="batch size for tmm_fast")
    args = ap.parse_args()

    set_seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["configs"]

    model = make_model_I(
        cfg.spec_dim, cfg.struc_dim, cfg.layers, cfg.d_model, cfg.d_ff, cfg.head_num, cfg.dropout
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    struc_word_dict = cfg.struc_word_dict
    struc_index_dict = cfg.struc_index_dict

    dev_struct = load_pickle(args.dev_struct)
    dev_spec = np.asarray(load_pickle(args.dev_spec), dtype=np.float32)

    N = len(dev_spec)
    spec_dim = dev_spec.shape[1]
    assert spec_dim == cfg.spec_dim, f"spec_dim mismatch: dev_spec={spec_dim}, cfg={cfg.spec_dim}"

    n_pts = int(round((args.lambda1 - args.lambda0) / args.step_um)) + 1
    wavelengths_um = np.linspace(args.lambda0, args.lambda1, n_pts)
    if spec_dim != 2 * n_pts:
        raise ValueError(f"spec_dim != 2*n_pts, got spec_dim={spec_dim}, 2*n_pts={2*n_pts}")

    mats_set = set()
    for seq in dev_struct:
        for s in seq:
            if "_" in s:
                mats_set.add(s.split("_", 1)[0])
    mats = sorted(list(mats_set))

    nk_dict_torch = load_nk_torch(args.nk_dir, mats, wavelengths_um)
    wl_m = torch.tensor(wavelengths_um * 1e-6, dtype=REAL_DTYPE, device=DEVICE)  # meters
    theta_rad = torch.tensor([0.0], dtype=REAL_DTYPE, device=DEVICE)

    idxs = list(range(N))
    if args.num_eval > 0 and args.num_eval < N:
        idxs = random.sample(idxs, args.num_eval)

    oracle_mats, oracle_thks, oracle_targets = [], [], []
    pred_mats, pred_thks, pred_targets = [], [], []
    pred_tokens_list = []
    pair_counter = Counter()

    # 先准备好要输出的 sample 信息（先不输出 mae）
    for j, ii in enumerate(idxs):
        spec_target = dev_spec[ii]
        gt_tokens = dev_struct[ii]

        mats_gt, thks_gt = parse_structure_tokens(gt_tokens)
        oracle_mats.append(mats_gt)
        oracle_thks.append(thks_gt)
        oracle_targets.append(spec_target)

        pred_tokens = greedy_decode(model, struc_index_dict, struc_word_dict, spec_target, args.max_len)
        pred_tokens_list.append(pred_tokens)
        pair_counter[infer_pair_name(pred_tokens)] += 1

        mats_pred, thks_pred = parse_structure_tokens(pred_tokens)
        pred_mats.append(mats_pred)
        pred_thks.append(thks_pred)
        pred_targets.append(spec_target)

    # batch 计算所有 mae
    oracle_mae_list = []
    pred_mae_list = []
    B = max(1, int(args.tmm_batch))

    for st in range(0, len(idxs), B):
        ed = min(len(idxs), st + B)

        spec_gt = calc_spec_tmmfast_batch(
            oracle_mats[st:ed], oracle_thks[st:ed],
            nk_dict_torch, wl_m, theta_rad, pol="s"
        )
        targets = np.asarray(oracle_targets[st:ed], dtype=np.float32)
        mae = np.mean(np.abs(spec_gt - targets), axis=1)
        oracle_mae_list.extend(mae.tolist())

        spec_pd = calc_spec_tmmfast_batch(
            pred_mats[st:ed], pred_thks[st:ed],
            nk_dict_torch, wl_m, theta_rad, pol="s"
        )
        targets2 = np.asarray(pred_targets[st:ed], dtype=np.float32)
        mae2 = np.mean(np.abs(spec_pd - targets2), axis=1)
        pred_mae_list.extend(mae2.tolist())

    # 现在再把前 print_k 个样例补上 mae 输出
    for j in range(min(args.print_k, len(idxs))):
        ii = idxs[j]
        gt_tokens = dev_struct[ii]
        pred_tokens = pred_tokens_list[j]
        print(f"\n---- sample {ii} ----")
        print(f"GT   pair: {infer_pair_name(gt_tokens)} | len={len(gt_tokens)}")
        print(f"PRED pair: {infer_pair_name(pred_tokens)} | len={len(pred_tokens)}")
        print("GT   head:", gt_tokens[:10], "..." if len(gt_tokens) > 10 else "")
        print("PRED head:", pred_tokens[:10], "..." if len(pred_tokens) > 10 else "")
        print(f"Oracle MAE: {oracle_mae_list[j]:.6f}")
        print(f"Pred   MAE: {pred_mae_list[j]:.6f}")

    oracle_mae = np.asarray(oracle_mae_list, dtype=np.float32)
    pred_mae = np.asarray(pred_mae_list, dtype=np.float32)

    print("\n==================== OVERALL ====================")
    print(f"eval samples: {len(idxs)}")

    print("\n[Oracle MAE] (GT -> TMM_FAST vs dev_spec)")
    print(f"  mean  : {oracle_mae.mean():.6f}")
    print(f"  median: {np.median(oracle_mae):.6f}")
    print(f"  p90   : {np.quantile(oracle_mae, 0.90):.6f}")

    print("\n[Pred MAE] (Pred -> TMM_FAST vs dev_spec)")
    print(f"  mean  : {pred_mae.mean():.6f}")
    print(f"  median: {np.median(pred_mae):.6f}")
    print(f"  p90   : {np.quantile(pred_mae, 0.90):.6f}")

    print("\n[PRED pair distribution] (top %d)" % args.pair_topk)
    total = sum(pair_counter.values())
    for k, v in pair_counter.most_common(args.pair_topk):
        print(f"  {k:20s} {v:6d} ({v/max(total,1):.3%})")

    print("==================================================")


if __name__ == "__main__":
    main()
