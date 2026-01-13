#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_dbr_smalltok_tmmfast.py

tmm_fast 版 eval（向量化 + 支持 batch）：
1) Oracle MAE：GT struct -> TMM_FAST -> spec vs dev_spec
2) Pred MAE：greedy decode -> TMM_FAST -> spec vs dev_spec
3) Pred pair 分布：快速看是否坍塌
4) 结构诊断：special/ood/不交替/厚度不在允许集合
5) Teacher-forcing：token-level NLL + top1 acc（排除 decode 的锅）
6) GT token OOV ratio：定位词表不一致
7) allowed thickness set：从 dev_struct 自动统计每个材料允许的厚度集合
   - pred_thk_not_allowed: 预测厚度不在允许集合（新数据集很关键）

用法同你原来一致。
"""

import os
import argparse
import random
import pickle as pkl
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from scipy.interpolate import interp1d

import tmm_fast  # 你的版本顶层导出 coh_tmm / inc_tmm
from core.models.transformer import make_model_I, subsequent_mask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COMPLEX_DTYPE = torch.complex64
REAL_DTYPE = torch.float32


# -------------------------
# utils
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
# nk load (numpy + torch)
# -------------------------
def load_nk_torch(nk_dir, materials, wavelengths_um) -> Dict[str, torch.Tensor]:
    """
    返回 nk_dict_torch[mat]: complex torch tensor, shape [W], on DEVICE
    """
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
# parse / pair
# -------------------------
def parse_structure_tokens(tokens):
    mats, thks = [], []
    for s in tokens:
        if "_" not in s:
            continue
        m, t = s.split("_", 1)
        try:
            mats.append(m)
            thks.append(int(round(float(t))))
        except Exception:
            continue
    return mats, thks

def infer_pair_name_from_tokens(tokens):
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
# greedy decode
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
        prob = model.generator(out[:, -1])  # log_softmax

        next_id = int(prob.argmax(dim=1).item())
        ys = torch.cat([ys, torch.ones(1, 1, dtype=torch.long, device=DEVICE).fill_(next_id)], dim=1)

        sym = struc_index_dict.get(next_id, "UNK")
        if sym == "EOS":
            break
        out_tokens.append(sym)

    return out_tokens


# -------------------------
# teacher forcing metrics
# -------------------------
@torch.no_grad()
def teacher_forcing_metrics(model, struc_word_dict, gt_tokens, spec_target,
                            pad_symbol="PAD", bos_symbol="BOS", eos_symbol="EOS",
                            ignore_special=True):
    pad_id = struc_word_dict.get(pad_symbol, 0)
    bos_id = struc_word_dict.get(bos_symbol, None)
    eos_id = struc_word_dict.get(eos_symbol, None)
    unk_id = struc_word_dict.get("UNK", None)

    if bos_id is None:
        return 0.0, 0, 0.0

    gt_ids = []
    for s in gt_tokens:
        if s in struc_word_dict:
            gt_ids.append(struc_word_dict[s])
        else:
            if unk_id is None:
                continue
            gt_ids.append(unk_id)

    if len(gt_ids) == 0:
        return 0.0, 0, 0.0

    tgt_in = [bos_id] + gt_ids
    tgt_y = gt_ids + ([eos_id] if eos_id is not None else [pad_id])

    tgt_in_t = torch.tensor(tgt_in, dtype=torch.long, device=DEVICE)[None, :]
    tgt_y_t  = torch.tensor(tgt_y,  dtype=torch.long, device=DEVICE)[None, :]

    spec_np = np.asarray(spec_target, dtype=np.float32)
    src = torch.from_numpy(spec_np).to(DEVICE)[None, None, :]
    src_mask = None

    L = tgt_in_t.size(1)
    tgt_mask = subsequent_mask(L).type_as(src.data).to(DEVICE)

    out = model(src, tgt_in_t, src_mask, tgt_mask)  # (1,L,d)
    logp = model.generator(out)                      # (1,L,vocab) log_softmax

    nll = -logp.gather(-1, tgt_y_t.unsqueeze(-1)).squeeze(-1)  # (1,L)
    pred = logp.argmax(dim=-1)
    correct = (pred == tgt_y_t).float()

    if ignore_special:
        ignore_ids = {pad_id, bos_id}
        if eos_id is not None:
            ignore_ids.add(eos_id)
        if unk_id is not None:
            ignore_ids.add(unk_id)
        mask = torch.ones_like(tgt_y_t, dtype=torch.bool)
        for iid in ignore_ids:
            mask &= (tgt_y_t != iid)
    else:
        mask = torch.ones_like(tgt_y_t, dtype=torch.bool)

    nll_sum = float(nll[mask].sum().item())
    n_tok = int(mask.sum().item())
    acc = float(correct[mask].sum().item()) / max(n_tok, 1)
    return nll_sum, n_tok, acc


# -------------------------
# OOV ratio
# -------------------------
def compute_oov_ratio(dev_struct, struc_word_dict):
    tot = 0
    oov = 0
    for seq in dev_struct:
        for t in seq:
            tot += 1
            if t not in struc_word_dict:
                oov += 1
    return oov, tot, (oov / max(tot, 1))


# -------------------------
# build allowed thickness set from dev_struct
# -------------------------
def build_allowed_thickness(dev_struct):
    allowed = defaultdict(set)  # mat -> set(thk_int)
    for seq in dev_struct:
        mats, thks = parse_structure_tokens(seq)
        for m, t in zip(mats, thks):
            allowed[m].add(int(t))
    return allowed


SPECIAL = {"UNK", "PAD", "BOS", "EOS"}

def check_structure(tokens, allowed_materials, allowed_thk_map, require_alternating=False):
    info = {}
    info["len_tokens"] = len(tokens)
    info["num_special"] = sum(1 for t in tokens if t in SPECIAL)
    info["has_special"] = info["num_special"] > 0

    mats, thks = parse_structure_tokens(tokens)
    info["num_layers_parsed"] = len(mats)
    if len(mats) == 0:
        info["reason"] = "empty_parsed"
        return False, info

    # alternating
    if len(mats) >= 2:
        info["is_alternating"] = all(mats[i] != mats[i - 1] for i in range(1, len(mats)))
    else:
        info["is_alternating"] = True

    # ood materials
    ood = [m for m in mats if m not in allowed_materials]
    info["num_ood_materials"] = len(ood)
    info["ood_materials_sample"] = sorted(set(ood))[:5]

    # thickness allowed check
    bad_thk = []
    for m, t in zip(mats, thks):
        if m in allowed_thk_map and int(t) not in allowed_thk_map[m]:
            bad_thk.append((m, int(t)))
    info["num_thk_not_allowed"] = len(bad_thk)
    info["thk_not_allowed_sample"] = bad_thk[:5]

    # basic thickness sanity
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
    if require_alternating and not info["is_alternating"]:
        info["reason"] = "not_alternating"
        return False, info
    if info["num_thk_not_allowed"] > 0:
        info["reason"] = "thk_not_allowed"
        return False, info
    if info["thk_min"] <= 0 or info["thk_max"] > 5000:
        info["reason"] = "bad_thickness_range"
        return False, info

    info["reason"] = "ok"
    return True, info


# -------------------------
# TMM_FAST: pack + calc
# -------------------------
def _pack_batch_to_tmm_fast(
    batch_mats: List[List[str]],
    batch_thks_nm: List[List[int]],
    nk_dict_torch: Dict[str, torch.Tensor],
    wl_m: torch.Tensor,  # [W]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回:
      n: [B, Lmax+2, W] complex
      d: [B, Lmax+2] real (meters), ends = inf
    """
    B = len(batch_mats)
    W = wl_m.shape[0]
    Lmax = max(len(x) for x in batch_mats)

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

        # nm -> m
        if L > 0:
            d[bi, 1:1 + L] = torch.tensor(thks, dtype=REAL_DTYPE, device=DEVICE) * 1e-9
            for li, m in enumerate(mats, start=1):
                n[bi, li, :] = nk_dict_torch[m]

        # pad：厚度=0，n 复制最后一层（零厚度对结果无影响）
        if L < Lmax and L > 0:
            n_pad = n[bi, L, :].clone()
            n[bi, 1 + L:1 + Lmax, :] = n_pad

    return n, d


@torch.no_grad()
def calc_RT_fast_batch(
    batch_mats: List[List[str]],
    batch_thks_nm: List[List[int]],
    nk_dict_torch: Dict[str, torch.Tensor],
    wl_m: torch.Tensor,       # [W]
    theta_rad: torch.Tensor,  # [A] (一般 [1])
    pol: str = "s",
) -> np.ndarray:
    """
    输出 spec: [B, 2W] float32 numpy, 按 [R..., T...]
    """
    n, d = _pack_batch_to_tmm_fast(batch_mats, batch_thks_nm, nk_dict_torch, wl_m)
    out = tmm_fast.coh_tmm(pol, n, d, theta_rad, wl_m)
    R = out["R"]
    T = out["T"]

    # 统一到 [B, W]
    if R.ndim == 3:
        R = R[:, 0, :]
        T = T[:, 0, :]
    elif R.ndim != 2:
        raise RuntimeError(f"Unexpected R shape: {tuple(R.shape)}")

    spec = torch.cat([R, T], dim=-1)  # [B, 2W]
    return spec.detach().cpu().float().numpy()


def calc_RT_fast_single(mats, thks_nm, nk_dict_torch, wl_m, theta_rad, pol="s") -> np.ndarray:
    spec = calc_RT_fast_batch([mats], [thks_nm], nk_dict_torch, wl_m, theta_rad, pol=pol)
    return spec[0]


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
    ap.add_argument("--num_eval", type=int, default=400, help="<=0 means all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--print_first_k", type=int, default=3)

    ap.add_argument("--require_alternating", action="store_true")
    ap.add_argument("--tf_include_special", action="store_true")

    # tmm_fast batch for oracle/pred
    ap.add_argument("--tmm_batch", type=int, default=128, help="batch size for tmm_fast calc")
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

    print("\n[CKPT cfg]")
    print("  spec_dim:", cfg.spec_dim)
    print("  struc_dim:", cfg.struc_dim)
    print("  layers/d_model/d_ff/heads:", cfg.layers, cfg.d_model, cfg.d_ff, cfg.head_num)

    print("\n[Vocab]")
    print("  vocab size:", len(struc_word_dict))
    for k in ["PAD", "BOS", "EOS", "UNK"]:
        print(f"  {k:4s}:", struc_word_dict.get(k, None))

    dev_struct = load_pickle(args.dev_struct)
    dev_spec = np.asarray(load_pickle(args.dev_spec), dtype=np.float32)

    N = len(dev_spec)
    spec_dim = dev_spec.shape[1]
    assert spec_dim == cfg.spec_dim, f"spec_dim mismatch: dev_spec={spec_dim}, cfg={cfg.spec_dim}"

    # wavelength grid must match
    n_pts = int(round((args.lambda1 - args.lambda0) / args.step_um)) + 1
    wavelengths_um = np.linspace(args.lambda0, args.lambda1, n_pts)
    if spec_dim != 2 * n_pts:
        print("[FATAL] spec_dim != 2*n_pts")
        print("  spec_dim:", spec_dim, "2*n_pts:", 2*n_pts, "n_pts:", n_pts)
        return

    # OOV
    oov, tot, ratio = compute_oov_ratio(dev_struct, struc_word_dict)
    print("\n[GT token OOV]")
    print(f"  OOV: {oov}/{tot} = {ratio:.4%}")

    # materials set from dev_struct
    mats_set = set()
    for seq in dev_struct:
        for s in seq:
            if "_" in s:
                mats_set.add(s.split("_", 1)[0])
    mats = sorted(list(mats_set))
    print("\nmaterials(from dev_struct):", mats)

    # allowed thickness per material (from dev_struct)
    allowed_thk = build_allowed_thickness(dev_struct)
    print("\n[Allowed thickness per material] (count only)")
    for m in mats:
        print(f"  {m:8s} unique_thk={len(allowed_thk[m])}")

    # nk torch
    nk_dict_torch = load_nk_torch(args.nk_dir, mats, wavelengths_um)

    # wl/theta for tmm_fast (SI)
    wl_m = torch.tensor(wavelengths_um * 1e-6, dtype=REAL_DTYPE, device=DEVICE)  # [W]
    theta_rad = torch.tensor([0.0], dtype=REAL_DTYPE, device=DEVICE)

    # choose subset
    idxs = list(range(N))
    if args.num_eval > 0 and args.num_eval < N:
        idxs = random.sample(idxs, args.num_eval)

    # accumulators
    mae_oracle, mae_pred = [], []
    bad_oracle, bad_pred = 0, 0

    pred_pair_counter = Counter()
    stats = Counter()

    tf_nll_sum, tf_n_tok = 0.0, 0
    tf_acc_sum, tf_batches = 0.0, 0
    ignore_special = (not args.tf_include_special)

    # ---------- 先把需要的 gt/pred 都准备好，后面 batch 算 tmm_fast ----------
    # oracle 需要的 batch
    oracle_batch_mats, oracle_batch_thks, oracle_batch_targets = [], [], []
    oracle_valid_flags = []  # True/False for each sample (parsed ok)

    # pred 需要的 batch（必须先 decode）
    pred_batch_mats, pred_batch_thks, pred_batch_targets = [], [], []
    pred_valid_flags = []  # True if structure check ok + parse ok
    pred_info_list = []    # for diagnostics

    # decode & teacher forcing & 结构检查
    for j, ii in enumerate(idxs):
        spec_target = dev_spec[ii]
        gt_tokens = dev_struct[ii]

        # teacher forcing
        nll_sum, n_tok, acc = teacher_forcing_metrics(
            model, struc_word_dict, gt_tokens, spec_target, ignore_special=ignore_special
        )
        tf_nll_sum += nll_sum
        tf_n_tok += n_tok
        tf_acc_sum += acc
        tf_batches += 1

        # oracle parse
        mats_gt, thks_gt = parse_structure_tokens(gt_tokens)
        if len(mats_gt) == 0:
            oracle_valid_flags.append(False)
        else:
            oracle_valid_flags.append(True)
            oracle_batch_mats.append(mats_gt)
            oracle_batch_thks.append(thks_gt)
            oracle_batch_targets.append(spec_target)

        # pred greedy decode
        pred_tokens = greedy_decode(model, struc_index_dict, struc_word_dict, spec_target, args.max_len)
        pred_pair = infer_pair_name_from_tokens(pred_tokens)
        pred_pair_counter[pred_pair] += 1

        ok, info = check_structure(
            pred_tokens,
            allowed_materials=set(mats),
            allowed_thk_map=allowed_thk,
            require_alternating=args.require_alternating
        )

        pred_info_list.append((ii, pred_tokens, ok, info, gt_tokens, pred_pair))

        if not ok:
            pred_valid_flags.append(False)
            stats[info["reason"]] += 1
        else:
            mats_pred, thks_pred = parse_structure_tokens(pred_tokens)
            if len(mats_pred) == 0:
                pred_valid_flags.append(False)
                stats["empty_parsed"] += 1
            else:
                pred_valid_flags.append(True)
                pred_batch_mats.append(mats_pred)
                pred_batch_thks.append(thks_pred)
                pred_batch_targets.append(spec_target)

        # print first k
        if j < args.print_first_k:
            print("\n---- sample", ii, "----")
            print("GT pair:", infer_pair_name_from_tokens(gt_tokens), "| len:", len(gt_tokens))
            print("PRED pair:", pred_pair, "| len:", len(pred_tokens))
            print("PRED head:", pred_tokens[:10])
            print("PRED check:", ok, info)

    # ---------- oracle: batch tmm_fast ----------
    if len(oracle_batch_mats) > 0:
        B = args.tmm_batch
        for st in range(0, len(oracle_batch_mats), B):
            ed = min(len(oracle_batch_mats), st + B)
            spec_pred = calc_RT_fast_batch(
                oracle_batch_mats[st:ed],
                oracle_batch_thks[st:ed],
                nk_dict_torch,
                wl_m,
                theta_rad,
                pol="s",
            )
            targets = np.asarray(oracle_batch_targets[st:ed], dtype=np.float32)
            mae = np.mean(np.abs(spec_pred - targets), axis=1)
            mae_oracle.extend(mae.tolist())
    bad_oracle = oracle_valid_flags.count(False)

    # ---------- pred: batch tmm_fast ----------
    if len(pred_batch_mats) > 0:
        B = args.tmm_batch
        for st in range(0, len(pred_batch_mats), B):
            ed = min(len(pred_batch_mats), st + B)
            spec_pred = calc_RT_fast_batch(
                pred_batch_mats[st:ed],
                pred_batch_thks[st:ed],
                nk_dict_torch,
                wl_m,
                theta_rad,
                pol="s",
            )
            targets = np.asarray(pred_batch_targets[st:ed], dtype=np.float32)
            mae = np.mean(np.abs(spec_pred - targets), axis=1)
            mae_pred.extend(mae.tolist())

    # pred failed count = total - valid_struct_ok_tmm
    bad_pred = len(idxs) - len(mae_pred)

    # report
    mae_oracle = np.asarray(mae_oracle, np.float32)
    mae_pred = np.asarray(mae_pred, np.float32)

    print("\n==================== REPORT ====================")
    print("eval samples:", len(idxs))

    print("\n[TEACHER FORCING]")
    print("  ignore_special:", ignore_special)
    print("  token count:", tf_n_tok)
    if tf_n_tok > 0:
        print("  avg NLL:", tf_nll_sum / tf_n_tok)
    if tf_batches > 0:
        print("  avg top1 acc (per-sample mean):", tf_acc_sum / tf_batches)

    print("\n[ORACLE]")
    print("  valid:", len(mae_oracle), "failed:", bad_oracle)
    if len(mae_oracle) > 0:
        print("  mean:", float(mae_oracle.mean()), "median:", float(np.median(mae_oracle)),
              "p90:", float(np.quantile(mae_oracle, 0.90)), "p99:", float(np.quantile(mae_oracle, 0.99)))

    print("\n[PRED greedy] (tmm_fast)")
    print("  valid:", len(mae_pred), "failed:", bad_pred)
    if len(mae_pred) > 0:
        print("  mean:", float(mae_pred.mean()), "median:", float(np.median(mae_pred)),
              "p90:", float(np.quantile(mae_pred, 0.90)), "p99:", float(np.quantile(mae_pred, 0.99)))

    print("\n[STRUCT DIAG reason counts]")
    for k, v in stats.most_common():
        print(f"  {k:20s}: {v}")

    print("\n[PRED pair distribution]")
    total_pred = sum(pred_pair_counter.values())
    for k, v in pred_pair_counter.most_common():
        print(f"  {k:20s} {v:6d} ({v/max(total_pred,1):.3%})")

    print("================================================")


if __name__ == "__main__":
    main()
