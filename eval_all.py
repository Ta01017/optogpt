#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_optogpt_universal_mae_tmmfast_mix.py

面向“大规模混合结构数据集”的通用 eval（tmm_fast batch）：
- Oracle MAE: GT -> TMM_FAST -> spec_pred vs dev_spec
- Pred   MAE: greedy decode -> TMM_FAST -> spec_pred vs dev_spec
- 可选 meta_dev.pkl:
  * 分组统计（按 family/type）
  * 分层抽样：保证每类结构在评估样本中的比例一致/可控
  * per-sample exit_medium（air / substrate），支持 Glass_Substrate 等

兼容：
- DBR / AR / FP / Multi-cavity / Random stack
- spec_type: R / T / R_T
- token: Material_ThicknessNm（如 TiO2_145）
- thickness: nm（脚本会 nm->m）

注意：
- 大规模时建议用 --num_eval + --stratified，避免随机抽样时 random 占比过大。
"""

import os
import argparse
import random
import pickle as pkl
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

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
    """Parse tokens like 'TiO2_145' into (mats, thks_nm). Ignore specials/unparsable."""
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
    """Human-readable printing: first two materials."""
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
def slice_spec_1d(spec: np.ndarray, spec_type: str) -> np.ndarray:
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
# nk loader (supports on-the-fly augmentation)
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
# tmm_fast packing (exit medium per batch)
# -------------------------
def _pack_batch_to_tmm_fast(
    batch_mats: List[List[str]],
    batch_thks_nm: List[List[float]],
    nk_dict_torch: Dict[str, torch.Tensor],
    wl_m: torch.Tensor,  # [W]
    exit_medium: Optional[str] = None,  # None -> air
    force_exit_k0: bool = False,        # for tmm_fast stability with lossy semi-inf
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    n: [B, Lmax+2, W], d: [B, Lmax+2]
    n[:,0]=air, n[:,-1]=air or substrate material.
    """
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
        if force_exit_k0:
            n_out = torch.real(nk_dict_torch[exit_medium]).to(REAL_DTYPE)
            n[:, -1, :] = n_out.to(COMPLEX_DTYPE) + 0.0j
        else:
            n[:, -1, :] = nk_dict_torch[exit_medium]

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
    spec_type: str = "R_T",
    force_exit_k0: bool = False,
) -> np.ndarray:
    """
    return spec: [B, D] where D depends on spec_type (R/T/R_T)
    """
    n, d = _pack_batch_to_tmm_fast(
        batch_mats, batch_thks_nm, nk_dict_torch, wl_m,
        exit_medium=exit_medium, force_exit_k0=force_exit_k0
    )
    out = tmm_fast.coh_tmm(pol, n, d, theta_rad, wl_m)

    R = out["R"]
    T = out["T"]
    if R.ndim == 3:
        R = R[:, 0, :]
        T = T[:, 0, :]
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
# decode
# -------------------------
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


# -------------------------
# meta/type + exit medium logic
# -------------------------
def infer_type_from_meta(meta: Optional[dict]) -> str:
    """
    尽量把你的混合数据归为可读类型：
    - 如果 meta["family"] 存在 -> 用它（比如 DBR/AR/FP/RANDOM）
    - 否则 fallback 到 meta["ar_family"] / meta["fp_family"] 等
    - 都没有 -> "UNKNOWN"
    """
    if meta is None:
        return "UNKNOWN"
    if "family" in meta and meta["family"]:
        return str(meta["family"])
    if "type" in meta and meta["type"]:
        return str(meta["type"])
    if "ar_family" in meta:
        return "AR"
    if "fp_family" in meta:
        return "FP"
    return "UNKNOWN"


def decide_exit_medium_for_sample(
    mats_gt: List[str],
    meta: Optional[dict],
    global_pref: Optional[str],
) -> Optional[str]:
    """
    per-sample exit medium:
    1) 若 meta 指明 exit_medium='substrate' 或给了 substrate 材料名 -> 用 substrate
    2) 若 GT 里出现 Glass_Substrate -> 用它
    3) 否则用 global_pref（'air'/'none'->None 或材料名）
    """
    # 1) meta-driven
    if meta is not None:
        # 常见写法：meta["exit_medium"] = "substrate" / "air" / "Glass_Substrate"
        em = meta.get("exit_medium", None)
        sub = meta.get("substrate", None)
        if isinstance(em, str):
            ems = em.strip().lower()
            if ems in ["air", "none", "null"]:
                return None
            if ems == "substrate":
                if isinstance(sub, str) and sub.strip():
                    return sub.strip()
                # 没给 substrate 名，就 fall back 到 Glass_Substrate
                return "Glass_Substrate" if "Glass_Substrate" in mats_gt else None
            # em 直接是材料名
            return em.strip()

        # 也可能只给 substrate
        if isinstance(sub, str) and sub.strip():
            return sub.strip()

    # 2) token contains substrate
    if "Glass_Substrate" in mats_gt:
        return "Glass_Substrate"

    # 3) global pref
    if global_pref is None:
        return None
    p = global_pref.strip()
    if p.lower() in ["air", "none", "null"]:
        return None
    return p


def stratified_sample_indices(
    types: List[str],
    total_n: int,
    mode: str = "proportional",  # proportional | equal
    seed: int = 0
) -> List[int]:
    """
    对大规模混合数据做分层抽样：
    - proportional：按原始类型分布采样 total_n
    - equal：每类采样相同数量（更利于对比）
    """
    rng = np.random.default_rng(seed)
    type2idx = defaultdict(list)
    for i, t in enumerate(types):
        type2idx[t].append(i)

    keys = sorted(type2idx.keys())
    if total_n <= 0:
        # all
        idxs = list(range(len(types)))
        rng.shuffle(idxs)
        return idxs

    if mode == "equal":
        k = len(keys)
        per = max(1, total_n // max(k, 1))
        out = []
        for t in keys:
            ids = type2idx[t]
            take = min(per, len(ids))
            out.extend(rng.choice(ids, size=take, replace=False).tolist())
        rng.shuffle(out)
        return out[:total_n]

    # proportional
    counts = np.array([len(type2idx[t]) for t in keys], dtype=np.float64)
    probs = counts / counts.sum()
    out = []
    # allocate
    alloc = np.floor(probs * total_n).astype(int)
    # fix remainder
    rem = total_n - int(alloc.sum())
    if rem > 0:
        extra = rng.choice(len(keys), size=rem, replace=True, p=probs)
        for j in extra:
            alloc[j] += 1

    for t, take in zip(keys, alloc.tolist()):
        ids = type2idx[t]
        if take <= 0:
            continue
        take = min(take, len(ids))
        out.extend(rng.choice(ids, size=take, replace=False).tolist())
    rng.shuffle(out)
    return out[:total_n]


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--dev_struct", type=str, required=True)
    ap.add_argument("--dev_spec", type=str, required=True)
    ap.add_argument("--dev_meta", type=str, default=None, help="optional meta_dev.pkl for type/exit-medium")

    ap.add_argument("--nk_dir", type=str, default="./dataset/data1")
    ap.add_argument("--lambda0", type=float, default=0.9)
    ap.add_argument("--lambda1", type=float, default=1.7)
    ap.add_argument("--step_um", type=float, default=0.005)

    ap.add_argument("--spec_type", type=str, default="R_T", help="R / T / R_T")
    ap.add_argument("--exit_medium", type=str, default=None,
                    help="global fallback exit medium: material name or 'air'. "
                         "If meta provides per-sample exit, it overrides.")
    ap.add_argument("--force_exit_k0", action="store_true",
                    help="force exit medium k=0 (use real(n)) for stability with semi-infinite lossy substrate.")
    ap.add_argument("--pol", type=str, default="s", help="s or p")

    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--num_eval", type=int, default=200, help="<=0 means all")
    ap.add_argument("--stratified", action="store_true", help="use stratified sampling by type (needs meta)")
    ap.add_argument("--strat_mode", type=str, default="proportional", help="proportional | equal (needs meta)")
    ap.add_argument("--print_k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pair_topk", type=int, default=10)
    ap.add_argument("--tmm_batch", type=int, default=128)
    args = ap.parse_args()

    set_seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["configs"]

    # model
    model = make_model_I(
        cfg.spec_dim, cfg.struc_dim, cfg.layers, cfg.d_model, cfg.d_ff, cfg.head_num, cfg.dropout
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    struc_word_dict = cfg.struc_word_dict
    struc_index_dict = cfg.struc_index_dict

    dev_struct = load_pickle(args.dev_struct)
    dev_spec_all = load_pickle(args.dev_spec)

    # dev_spec 可能是 list[list[float]]，转 np.float32（大规模注意内存：建议先 num_eval 小抽样）
    dev_spec_all = np.asarray(dev_spec_all, dtype=np.float32)

    N = len(dev_spec_all)
    if N != len(dev_struct):
        raise ValueError(f"len(dev_spec)={N} != len(dev_struct)={len(dev_struct)}")

    dev_meta = None
    types = ["UNKNOWN"] * N
    if args.dev_meta is not None and os.path.exists(args.dev_meta):
        dev_meta = load_pickle(args.dev_meta)
        if len(dev_meta) != N:
            raise ValueError(f"len(dev_meta)={len(dev_meta)} != N={N}")
        types = [infer_type_from_meta(dev_meta[i]) for i in range(N)]

    # Wavelength grid
    n_pts = int(round((args.lambda1 - args.lambda0) / args.step_um)) + 1
    wavelengths_um = np.linspace(args.lambda0, args.lambda1, n_pts)

    # Slice dev_spec by spec_type
    #（大规模避免 python for 逐条太慢：这里仍然 O(N)，但你通常会 num_eval；若全量可改成向量化）
    dev_spec = np.stack([slice_spec_1d(dev_spec_all[i], args.spec_type) for i in range(N)], axis=0)
    spec_dim = dev_spec.shape[1]
    if spec_dim != cfg.spec_dim:
        raise ValueError(f"spec_dim mismatch after slicing: dev_spec={spec_dim}, cfg={cfg.spec_dim} "
                         f"(train spec_type must match eval spec_type)")

    # pick indices
    if args.stratified and dev_meta is None:
        raise ValueError("--stratified requires --dev_meta meta_dev.pkl")

    if args.stratified and args.num_eval > 0:
        idxs = stratified_sample_indices(types, args.num_eval, mode=args.strat_mode, seed=args.seed)
    else:
        idxs = list(range(N))
        if args.num_eval > 0 and args.num_eval < N:
            idxs = random.sample(idxs, args.num_eval)

    # Collect GT materials first (only in selected idxs to reduce nk load)
    mats_in_data = set()
    for ii in idxs:
        for s in dev_struct[ii]:
            if "_" in s:
                mats_in_data.add(s.split("_", 1)[0])
    mats_in_data = sorted(list(mats_in_data))

    # nk (start with GT mats; pred new mats will be loaded on the fly)
    nk_dict_torch = load_nk_torch(args.nk_dir, mats_in_data, wavelengths_um)

    wl_m = torch.tensor(wavelengths_um * 1e-6, dtype=REAL_DTYPE, device=DEVICE)
    theta_rad = torch.tensor([0.0], dtype=REAL_DTYPE, device=DEVICE)

    # containers
    oracle_mae = []
    pred_mae = []
    pred_pair_counter = Counter()
    pred_len_counter = Counter()
    pred_tok_counter = Counter()

    # per-type containers
    per_type_oracle = defaultdict(list)
    per_type_pred = defaultdict(list)
    per_type_len = defaultdict(list)
    per_type_pair = defaultdict(Counter)

    # sample cache for printing
    sample_cache = []

    # on-the-fly nk load for unseen predicted mats
    def ensure_nk(materials: List[str]):
        new_mats = [m for m in materials if m not in nk_dict_torch]
        if new_mats:
            nk_new = load_nk_torch(args.nk_dir, sorted(list(set(new_mats))), wavelengths_um)
            nk_dict_torch.update(nk_new)

    # batching: 为了支持“每样本 exit_medium 可能不同”，我们把 batch 内按 exit_medium 分桶
    B = max(1, int(args.tmm_batch))

    # decode & evaluate in chunks
    for st in range(0, len(idxs), B):
        chunk = idxs[st: st + B]

        # ---- build chunk data ----
        gt_mats_list, gt_thks_list = [], []
        pred_mats_list, pred_thks_list = [], []
        tgt_specs = []
        exit_list = []
        type_list = []
        gt_tokens_list = []
        pred_tokens_list = []

        for ii in chunk:
            tgt = dev_spec[ii]
            gt_tokens = dev_struct[ii]
            meta_i = dev_meta[ii] if dev_meta is not None else None
            tname = types[ii] if dev_meta is not None else "UNKNOWN"

            mats_gt, thks_gt = parse_structure_tokens(gt_tokens)

            # decode pred
            pred_tokens = greedy_decode(model, struc_index_dict, struc_word_dict, tgt, args.max_len)
            mats_pd, thks_pd = parse_structure_tokens(pred_tokens)

            # decide exit medium per sample
            em = decide_exit_medium_for_sample(mats_gt, meta_i, args.exit_medium)

            gt_mats_list.append(mats_gt)
            gt_thks_list.append(thks_gt)
            pred_mats_list.append(mats_pd)
            pred_thks_list.append(thks_pd)
            tgt_specs.append(tgt)
            exit_list.append(em)
            type_list.append(tname)
            gt_tokens_list.append(gt_tokens)
            pred_tokens_list.append(pred_tokens)

            # counters (pred)
            pn = infer_pair_name(pred_tokens)
            pred_pair_counter[pn] += 1
            pred_len_counter[len(pred_tokens)] += 1
            per_type_pair[tname][pn] += 1
            for tok in pred_tokens:
                if "_" in tok:
                    pred_tok_counter[tok] += 1

        tgt_specs = np.asarray(tgt_specs, dtype=np.float32)

        # ensure nk for predicted materials in this chunk
        ensure_nk([m for mats in pred_mats_list for m in mats])
        # also ensure nk for exit media if any
        ensure_nk([em for em in exit_list if em is not None])

        # ---- evaluate oracle/pred grouped by exit_medium ----
        # group indices in chunk by exit_medium (air=None vs substrate material)
        group_map = defaultdict(list)
        for j, em in enumerate(exit_list):
            group_map[em].append(j)

        for em, js in group_map.items():
            # oracle
            spec_or = calc_spec_tmmfast_batch(
                [gt_mats_list[j] for j in js],
                [gt_thks_list[j] for j in js],
                nk_dict_torch, wl_m, theta_rad,
                pol=args.pol, exit_medium=em,
                spec_type=args.spec_type,
                force_exit_k0=args.force_exit_k0,
            )
            # pred
            spec_pd = calc_spec_tmmfast_batch(
                [pred_mats_list[j] for j in js],
                [pred_thks_list[j] for j in js],
                nk_dict_torch, wl_m, theta_rad,
                pol=args.pol, exit_medium=em,
                spec_type=args.spec_type,
                force_exit_k0=args.force_exit_k0,
            )

            tgt_js = tgt_specs[js]
            mae_or = np.mean(np.abs(spec_or - tgt_js), axis=1)
            mae_pd = np.mean(np.abs(spec_pd - tgt_js), axis=1)

            # write back
            for k, j in enumerate(js):
                oracle_mae.append(float(mae_or[k]))
                pred_mae.append(float(mae_pd[k]))

                tname = type_list[j]
                per_type_oracle[tname].append(float(mae_or[k]))
                per_type_pred[tname].append(float(mae_pd[k]))
                per_type_len[tname].append(len(pred_tokens_list[j]))

        # ---- cache samples for printing (first print_k) ----
        for j in range(len(chunk)):
            if len(sample_cache) >= args.print_k:
                break
            sample_cache.append(dict(
                idx=chunk[j],
                type=type_list[j],
                exit=exit_list[j] if exit_list[j] is not None else "air",
                gt=gt_tokens_list[j],
                pred=pred_tokens_list[j],
            ))

    oracle_mae = np.asarray(oracle_mae, dtype=np.float32)
    pred_mae = np.asarray(pred_mae, dtype=np.float32)

    # ---- print samples ----
    for it, s in enumerate(sample_cache):
        ii = s["idx"]
        gt_tokens = s["gt"]
        pred_tokens = s["pred"]
        print(f"\n---- sample {ii} ----")
        print(f"type={s['type']} | exit={s['exit']}")
        print(f"GT   pair: {infer_pair_name(gt_tokens)} | len={len(gt_tokens)}")
        print(f"PRED pair: {infer_pair_name(pred_tokens)} | len={len(pred_tokens)}")
        print("GT   head:", gt_tokens[:10], "..." if len(gt_tokens) > 10 else "")
        print("PRED head:", pred_tokens[:10], "..." if len(pred_tokens) > 10 else "")
        print(f"Oracle MAE: {oracle_mae[it]:.6f}")
        print(f"Pred   MAE: {pred_mae[it]:.6f}")

    # ---- overall ----
    print("\n==================== OVERALL ====================")
    print(f"eval samples: {len(oracle_mae)} (from total N={N})")
    print(f"spec_type={args.spec_type} | pol={args.pol} | force_exit_k0={args.force_exit_k0}")
    print(f"loaded_materials={len(nk_dict_torch)}")
    if dev_meta is not None:
        type_cnt = Counter([types[i] for i in idxs])
        print(f"types_in_eval={len(type_cnt)} | stratified={args.stratified} mode={args.strat_mode}")
        for k, v in type_cnt.most_common(20):
            print(f"  type={k:10s}  {v:6d} ({v/len(idxs):.2%})")

    print("\n[Oracle MAE] (GT -> TMM_FAST vs dev_spec)")
    print(f"  mean  : {oracle_mae.mean():.6f}")
    print(f"  median: {np.median(oracle_mae):.6f}")
    print(f"  p90   : {np.quantile(oracle_mae, 0.90):.6f}")

    print("\n[Pred MAE] (Pred(greedy) -> TMM_FAST vs dev_spec)")
    print(f"  mean  : {pred_mae.mean():.6f}")
    print(f"  median: {np.median(pred_mae):.6f}")
    print(f"  p90   : {np.quantile(pred_mae, 0.90):.6f}")

    # ---- per-type ----
    if dev_meta is not None:
        print("\n==================== PER-TYPE ====================")
        keys = sorted(per_type_pred.keys())
        for t in keys:
            om = np.asarray(per_type_oracle[t], dtype=np.float32)
            pm = np.asarray(per_type_pred[t], dtype=np.float32)
            ln = per_type_len[t]
            if len(pm) == 0:
                continue
            print(f"\n[type={t}] n={len(pm)}")
            print(f"  Oracle mean={om.mean():.6f} | Pred mean={pm.mean():.6f} | Pred median={np.median(pm):.6f} | Pred p90={np.quantile(pm,0.90):.6f}")
            print(f"  Pred length: min={min(ln)} mean={float(np.mean(ln)):.3f} max={max(ln)}")
            top_pairs = per_type_pair[t].most_common(5)
            if top_pairs:
                print("  Top pred pairs:", ", ".join([f"{k}({v})" for k, v in top_pairs]))

    # ---- distributions ----
    print("\n[PRED pair distribution] (top %d)" % args.pair_topk)
    total = sum(pred_pair_counter.values())
    for k, v in pred_pair_counter.most_common(args.pair_topk):
        print(f"  {k:20s} {v:6d} ({v/max(total,1):.3%})")

    print("\n[PRED length distribution] (top 10)")
    total2 = sum(pred_len_counter.values())
    for k, v in pred_len_counter.most_common(10):
        print(f"  len={k:3d} {v:6d} ({v/max(total2,1):.3%})")

    print("\n[Top predicted tokens] (top 10)")
    tot_tok = sum(pred_tok_counter.values())
    for k, v in pred_tok_counter.most_common(10):
        print(f"  {k:18s} {v:7d} ({v/max(tot_tok,1):.3%})")

    print("==================================================")


if __name__ == "__main__":
    main()

"""
python eval_all.py \
  --ckpt saved_models/optogpt/mix_1m/model_inverse_best.pt \
  --dev_struct ./dataset/mix/Structure_dev.pkl \
  --dev_spec   ./dataset/mix/Spectrum_dev.pkl \
  --dev_meta   ./dataset/mix/meta_dev.pkl \
  --nk_dir     ./dataset/data1 \
  --spec_type  R_T \
  --max_len 22 \
  --num_eval 2000 \
  --stratified --strat_mode proportional \
  --tmm_batch 128 \
  --print_k 10
"""