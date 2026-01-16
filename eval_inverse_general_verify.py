#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_inverse_general_verify.py

通用验证 eval（适配你 OptoGPT/Transformer inverse：spec -> structure），用于“验证是不是多解导致学不到”。

核心功能（你要的验证）：
A) Teacher-forcing 验证（token-level）
   - next-token NLL / CE
   - top1 accuracy
   - token length statistics（真实/预测）

B) Decode 验证（结构->光谱一致性）
   - Greedy decode -> tmm_fast -> MAE(spec_pred, spec_gt)
   - Sampling decode (top-p/top-k/temperature) -> tmm_fast -> MAE
   - Best-of-N：对同一条 spec 采样 N 次，取最小 MAE（检测“多解+greedy坍塌”）

C) Collapse/多样性统计
   - 预测 token 分布 topK
   - 预测“材料pattern”（只看材料序列，不看厚度）topK
   - EOS length 分布（是否生成太长/太短）

D) （可选）多解指数：光谱近邻结构分散度
   - 在 dev_spec 上抽样 M 条，做最近邻（L2），看“谱很近但结构差很大”的比例
   - 用于直接证明“多解”

依赖：
- tmm_fast
- core.models.transformer.make_model_I / subsequent_mask
- 你的 ckpt 里包含 configs（同你 DBR eval 的存法）

用法例子：
python eval_inverse_general_verify.py \
  --ckpt saved_models/optogpt/ar_smalltoken/best.pt \
  --dev_struct ./dataset/ar_smalltoken_gpu/Structure_dev.pkl \
  --dev_spec   ./dataset/ar_smalltoken_gpu/Spectrum_dev.pkl \
  --nk_dir ./dataset/data1 \
  --lambda0 0.9 --lambda1 1.7 --step_um 0.005 \
  --exit_medium substrate --substrate_name Glass_Substrate \
  --strategy both --num_eval 500 --print_k 5 \
  --sample_N 20 --top_p 0.95 --top_k 50 --temperature 1.1 \
  --tmm_batch 128 \
  --nn_check 200 --nn_k 5

重要参数：
--exit_medium: air / substrate（AR 推荐 substrate；DBR/FP 通常 air）
--strategy: greedy / sample / both
--sample_N: best-of-N 的 N
--nn_check: 开启多解指数检查（>0 才做）

"""

import os
import math
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

# -------------------------
# Global dtypes / device
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMPLEX_DTYPE = torch.complex64
REAL_DTYPE = torch.float32


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pkl.load(f)


def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


def parse_structure_tokens(tokens: List[str]) -> Tuple[List[str], List[float]]:
    """tokens: ["TiO2_120", "SiO2_180", ...] -> mats, thks_nm(float)"""
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


def tokens_to_material_pattern(tokens: List[str]) -> str:
    """只看材料序列，不看厚度，用于统计坍塌（例如: TiO2/SiO2/TiO2/SiO2）"""
    mats = []
    for s in tokens:
        if "_" in s:
            mats.append(s.split("_", 1)[0])
    if len(mats) == 0:
        return "EMPTY"
    # 用 '/' 拼接
    return "/".join(mats)


def infer_pair_name(tokens: List[str]) -> str:
    """取前两个材料做 pair name（兼容你之前的统计风格）"""
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
# nk loader
# -------------------------
def load_nk_torch(nk_dir: str, materials: List[str], wavelengths_um: np.ndarray) -> Dict[str, torch.Tensor]:
    nk: Dict[str, torch.Tensor] = {}
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
# tmm_fast pack + spec
# -------------------------
def _pack_batch_to_tmm_fast(
    batch_mats: List[List[str]],
    batch_thks_nm: List[List[float]],
    nk_dict_torch: Dict[str, torch.Tensor],
    wl_m: torch.Tensor,  # [W]
    exit_medium: str = "air",
    substrate_name: str = "Glass_Substrate",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return:
      n: [B, Lmax+2, W] complex
      d: [B, Lmax+2] real (meters), inf at ends
    Medium convention:
      n[:,0] = air (1.0)
      n[:,-1] = air or substrate nk
    """
    B = len(batch_mats)
    W = wl_m.shape[0]
    Lmax = max((len(x) for x in batch_mats), default=0)

    d = torch.zeros((B, Lmax + 2), dtype=REAL_DTYPE, device=DEVICE)
    d[:, 0] = float("inf")
    d[:, -1] = float("inf")

    n = torch.empty((B, Lmax + 2, W), dtype=COMPLEX_DTYPE, device=DEVICE)
    n[:, 0, :] = (1.0 + 0.0j)

    if exit_medium == "air":
        n[:, -1, :] = (1.0 + 0.0j)
    elif exit_medium == "substrate":
        if substrate_name not in nk_dict_torch:
            raise KeyError(f"substrate_name='{substrate_name}' not found in nk_dict_torch.")
        n[:, -1, :] = nk_dict_torch[substrate_name]
    else:
        raise ValueError(f"exit_medium must be 'air' or 'substrate', got {exit_medium}")

    for bi in range(B):
        mats = batch_mats[bi]
        thks = batch_thks_nm[bi]
        L = len(mats)
        if L > 0:
            d[bi, 1:1 + L] = torch.tensor(thks, dtype=REAL_DTYPE, device=DEVICE) * 1e-9  # nm->m
            for li, m in enumerate(mats, start=1):
                n[bi, li, :] = nk_dict_torch[m]

        # pad: thickness=0, n copy last real layer (zero thickness -> no effect)
        if L < Lmax and L > 0:
            n_pad = n[bi, L, :].clone()
            n[bi, 1 + L:1 + Lmax, :] = n_pad
        elif L == 0 and Lmax > 0:
            # empty structure -> set pad to air
            n[bi, 1:1 + Lmax, :] = (1.0 + 0.0j)

    return n, d


@torch.no_grad()
def calc_spec_tmmfast_batch(
    batch_mats: List[List[str]],
    batch_thks_nm: List[List[float]],
    nk_dict_torch: Dict[str, torch.Tensor],
    wl_m: torch.Tensor,
    theta_rad: torch.Tensor,
    pol: str = "s",
    exit_medium: str = "air",
    substrate_name: str = "Glass_Substrate",
) -> np.ndarray:
    """
    Return spec: [B, 2W] = [R..., T...]
    """
    n, d = _pack_batch_to_tmm_fast(batch_mats, batch_thks_nm, nk_dict_torch, wl_m,
                                  exit_medium=exit_medium, substrate_name=substrate_name)
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


# -------------------------
# decoding: greedy / sampling
# -------------------------
def _get_bos_eos_ids(struc_word_dict: Dict[str, int]) -> Tuple[int, Optional[int]]:
    # 兼容你的字典（BOS/EOS）
    bos_id = struc_word_dict.get("BOS", None)
    eos_id = struc_word_dict.get("EOS", None)
    if bos_id is None:
        raise KeyError("struc_word_dict missing 'BOS'")
    return int(bos_id), (int(eos_id) if eos_id is not None else None)


@torch.no_grad()
def greedy_decode_tokens(
    model,
    struc_index_dict: Dict[int, str],
    struc_word_dict: Dict[str, int],
    spec_target: np.ndarray,
    max_len: int,
) -> List[str]:
    bos_id, eos_id = _get_bos_eos_ids(struc_word_dict)
    ys = torch.ones(1, 1, dtype=torch.long, device=DEVICE).fill_(bos_id)

    src = torch.from_numpy(np.asarray(spec_target, dtype=np.float32)).to(DEVICE)[None, None, :]
    src_mask = None

    out_tokens = []
    for _ in range(max_len - 1):
        trg_mask = Variable(subsequent_mask(ys.size(1)).type_as(src.data)).to(DEVICE)
        out = model(src, Variable(ys), src_mask, trg_mask)
        prob = model.generator(out[:, -1])  # [1, V]

        next_id = int(prob.argmax(dim=1).item())
        ys = torch.cat([ys, torch.ones(1, 1, dtype=torch.long, device=DEVICE).fill_(next_id)], dim=1)

        sym = struc_index_dict.get(next_id, "UNK")
        if sym == "EOS":
            break
        out_tokens.append(sym)

    return out_tokens


def _sample_from_logits(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0, temperature: float = 1.0) -> int:
    """
    logits: [V]
    returns sampled token id (int)
    """
    if temperature <= 0:
        temperature = 1.0
    logits = logits / float(temperature)

    # top_k filter
    if top_k is not None and top_k > 0 and top_k < logits.numel():
        values, idx = torch.topk(logits, top_k)
        mask = torch.full_like(logits, float("-inf"))
        mask[idx] = logits[idx]
        logits = mask

    # top_p filter
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)

        cutoff = cum > top_p
        # 保留第一个超过 top_p 的位置
        cutoff[0] = False
        sorted_logits[cutoff] = float("-inf")

        # map back
        new_logits = torch.full_like(logits, float("-inf"))
        new_logits[sorted_idx] = sorted_logits
        logits = new_logits

    probs = torch.softmax(logits, dim=-1)
    # 防止 nan
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = probs / probs.sum()

    sampled = torch.multinomial(probs, num_samples=1).item()
    return int(sampled)


@torch.no_grad()
def sample_decode_tokens(
    model,
    struc_index_dict: Dict[int, str],
    struc_word_dict: Dict[str, int],
    spec_target: np.ndarray,
    max_len: int,
    top_k: int = 0,
    top_p: float = 1.0,
    temperature: float = 1.0,
) -> List[str]:
    bos_id, eos_id = _get_bos_eos_ids(struc_word_dict)
    ys = torch.ones(1, 1, dtype=torch.long, device=DEVICE).fill_(bos_id)

    src = torch.from_numpy(np.asarray(spec_target, dtype=np.float32)).to(DEVICE)[None, None, :]
    src_mask = None

    out_tokens = []
    for _ in range(max_len - 1):
        trg_mask = Variable(subsequent_mask(ys.size(1)).type_as(src.data)).to(DEVICE)
        out = model(src, Variable(ys), src_mask, trg_mask)
        logits = model.generator(out[:, -1]).squeeze(0)  # [V]

        next_id = _sample_from_logits(logits, top_k=top_k, top_p=top_p, temperature=temperature)
        ys = torch.cat([ys, torch.ones(1, 1, dtype=torch.long, device=DEVICE).fill_(next_id)], dim=1)

        sym = struc_index_dict.get(next_id, "UNK")
        if sym == "EOS":
            break
        out_tokens.append(sym)

    return out_tokens


# -------------------------
# Teacher forcing evaluation
# -------------------------
@torch.no_grad()
def teacher_forcing_metrics(
    model,
    dev_spec: np.ndarray,                # [N, spec_dim]
    dev_struct: List[List[str]],         # list of token strings (no BOS/EOS)
    struc_word_dict: Dict[str, int],
    pad_symbol: str = "PAD",
    bos_symbol: str = "BOS",
    eos_symbol: str = "EOS",
    max_len: int = 64,
    batch_size: int = 64,
) -> Dict[str, float]:
    """
    计算 next-token NLL/CE + top1 acc（teacher forcing）
    说明：
      - 需要把 GT token 映射成 id，并加 BOS/EOS
      - 超过 max_len 会截断（保持与你训练一致）
    """
    pad_id = int(struc_word_dict.get(pad_symbol, 0))
    bos_id = int(struc_word_dict[bos_symbol])
    eos_id = int(struc_word_dict[eos_symbol])

    V = len(struc_word_dict)

    def tokens_to_ids(seq_tokens: List[str]) -> List[int]:
        ids = [bos_id]
        for t in seq_tokens:
            ids.append(int(struc_word_dict.get(t, struc_word_dict.get("UNK", pad_id))))
        ids.append(eos_id)
        # truncate / pad to max_len
        if len(ids) > max_len:
            ids = ids[:max_len]
            # 保证最后一个是 EOS（更稳定）
            ids[-1] = eos_id
        return ids

    all_ids = [tokens_to_ids(seq) for seq in dev_struct]
    N = len(all_ids)

    total_tokens = 0
    correct = 0
    total_nll = 0.0

    for st in range(0, N, batch_size):
        ed = min(N, st + batch_size)
        B = ed - st
        # prepare src
        src = torch.from_numpy(dev_spec[st:ed].astype(np.float32)).to(DEVICE)[:, None, :]  # [B,1,spec]
        src_mask = None

        # prepare tgt ids (pad)
        tgt_lens = [len(x) for x in all_ids[st:ed]]
        L = max(tgt_lens)
        L = min(L, max_len)
        tgt = torch.full((B, L), pad_id, dtype=torch.long, device=DEVICE)
        for i, ids in enumerate(all_ids[st:ed]):
            ids = ids[:L]
            tgt[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=DEVICE)

        # Teacher forcing: input = tgt[:,:-1], predict next = tgt[:,1:]
        inp = tgt[:, :-1]
        gold = tgt[:, 1:]

        trg_mask = subsequent_mask(inp.size(1)).type_as(src.data).to(DEVICE)  # [1,L-1,L-1]
        out = model(src, inp, src_mask, trg_mask)                            # [B,L-1,d]
        logits = model.generator(out)                                        # [B,L-1,V]

        # flatten
        logits_f = logits.reshape(-1, V)
        gold_f = gold.reshape(-1)

        # mask pad
        mask = gold_f != pad_id
        if mask.sum().item() == 0:
            continue

        gold_m = gold_f[mask]
        logits_m = logits_f[mask]

        logp = torch.log_softmax(logits_m, dim=-1)
        nll = -logp.gather(1, gold_m[:, None]).squeeze(1)  # [T]
        total_nll += float(nll.sum().item())

        pred = torch.argmax(logits_m, dim=-1)
        correct += int((pred == gold_m).sum().item())
        total_tokens += int(gold_m.numel())

    ce = total_nll / max(total_tokens, 1)
    acc = correct / max(total_tokens, 1)

    return {
        "tf_token_ce": float(ce),
        "tf_token_acc": float(acc),
        "tf_tokens": int(total_tokens),
    }


# -------------------------
# Nearest-neighbor ambiguity check
# -------------------------
def ambiguity_nn_check(
    dev_spec: np.ndarray,
    dev_struct: List[List[str]],
    check_n: int = 200,
    nn_k: int = 5,
    seed: int = 0,
) -> Dict[str, float]:
    """
    多解指数验证：
      抽样 check_n 条样本，对每条找最近 nn_k 个邻居（L2），统计：
      - 邻居平均谱距离
      - 邻居结构完全相同的比例
      - 邻居材料pattern相同的比例
    """
    rng = np.random.default_rng(seed)
    N = len(dev_spec)
    if check_n <= 0:
        return {}
    check_n = min(check_n, N)

    idxs = rng.choice(N, size=check_n, replace=False)

    same_struct = 0
    same_pat = 0
    total_neighbors = 0
    mean_nn_dist = []

    # 预计算 patterns
    patterns = [tokens_to_material_pattern(s) for s in dev_struct]

    # 用 brute-force（check_n 小就行）
    for ii in idxs:
        s = dev_spec[ii]
        # L2 distances to all
        d = np.linalg.norm(dev_spec - s[None, :], axis=1)
        # exclude itself
        d[ii] = np.inf
        nn = np.argpartition(d, nn_k)[:nn_k]
        mean_nn_dist.append(float(np.mean(d[nn])))

        for j in nn:
            total_neighbors += 1
            if dev_struct[j] == dev_struct[ii]:
                same_struct += 1
            if patterns[j] == patterns[ii]:
                same_pat += 1

    return {
        "nn_check_n": int(check_n),
        "nn_k": int(nn_k),
        "nn_mean_dist": float(np.mean(mean_nn_dist)) if len(mean_nn_dist) else float("nan"),
        "nn_same_structure_rate": float(same_struct / max(total_neighbors, 1)),
        "nn_same_pattern_rate": float(same_pat / max(total_neighbors, 1)),
    }


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--dev_struct", type=str, required=True)
    ap.add_argument("--dev_spec", type=str, required=True)
    ap.add_argument("--nk_dir", type=str, default="./dataset/data1")

    ap.add_argument("--lambda0", type=float, default=0.9)
    ap.add_argument("--lambda1", type=float, default=1.7)
    ap.add_argument("--step_um", type=float, default=0.005)

    ap.add_argument("--pol", type=str, default="s", choices=["s", "p"])
    ap.add_argument("--exit_medium", type=str, default="air", choices=["air", "substrate"])
    ap.add_argument("--substrate_name", type=str, default="Glass_Substrate")

    ap.add_argument("--max_len", type=int, default=22)
    ap.add_argument("--num_eval", type=int, default=500, help="<=0 means all")
    ap.add_argument("--print_k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--strategy", type=str, default="both", choices=["greedy", "sample", "both"])
    ap.add_argument("--sample_N", type=int, default=20, help="best-of-N sampling")
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=1.1)

    ap.add_argument("--tmm_batch", type=int, default=128)
    ap.add_argument("--pair_topk", type=int, default=10)

    ap.add_argument("--tf_batch", type=int, default=128, help="teacher-forcing batch")
    ap.add_argument("--do_tf", action="store_true", help="enable teacher forcing metrics")

    ap.add_argument("--nn_check", type=int, default=0, help=">0 enable ambiguity NN check, number of samples")
    ap.add_argument("--nn_k", type=int, default=5, help="NN k neighbors")

    args = ap.parse_args()
    set_seed(args.seed)

    # ---- load ckpt + model ----
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["configs"]

    model = make_model_I(
        cfg.spec_dim, cfg.struc_dim, cfg.layers, cfg.d_model, cfg.d_ff, cfg.head_num, cfg.dropout
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    struc_word_dict = cfg.struc_word_dict
    struc_index_dict = cfg.struc_index_dict

    # ---- load data ----
    dev_struct = load_pickle(args.dev_struct)
    dev_spec = np.asarray(load_pickle(args.dev_spec), dtype=np.float32)

    N_all = len(dev_spec)
    spec_dim = dev_spec.shape[1]
    if spec_dim != cfg.spec_dim:
        raise ValueError(f"spec_dim mismatch: dev_spec={spec_dim}, cfg={cfg.spec_dim}")

    # ---- wavelength grid consistency ----
    n_pts = int(round((args.lambda1 - args.lambda0) / args.step_um)) + 1
    wavelengths_um = np.linspace(args.lambda0, args.lambda1, n_pts)
    if spec_dim != 2 * n_pts:
        raise ValueError(f"spec_dim != 2*n_pts, got spec_dim={spec_dim}, 2*n_pts={2*n_pts}")

    wl_m = torch.tensor(wavelengths_um * 1e-6, dtype=REAL_DTYPE, device=DEVICE)
    theta_rad = torch.tensor([0.0], dtype=REAL_DTYPE, device=DEVICE)

    # ---- infer materials used in dev_struct (plus substrate if needed) ----
    mats_set = set()
    for seq in dev_struct:
        for s in seq:
            if "_" in s:
                mats_set.add(s.split("_", 1)[0])
    if args.exit_medium == "substrate":
        mats_set.add(args.substrate_name)
    mats = sorted(list(mats_set))

    nk_dict_torch = load_nk_torch(args.nk_dir, mats, wavelengths_um)

    # ---- choose eval subset ----
    idxs = list(range(N_all))
    if args.num_eval > 0 and args.num_eval < N_all:
        idxs = random.sample(idxs, args.num_eval)

    print("==================================================")
    print(f"DEVICE={DEVICE} | pol={args.pol} | exit_medium={args.exit_medium}")
    print(f"eval_samples={len(idxs)} / {N_all} | max_len={args.max_len}")
    print(f"loaded_materials={len(mats)}")
    if args.exit_medium == "substrate":
        print(f"substrate_name={args.substrate_name}")
    print("==================================================")

    # ---- Optional: ambiguity NN check (direct proof of multi-solution) ----
    if args.nn_check and args.nn_check > 0:
        nnr = ambiguity_nn_check(dev_spec, dev_struct, check_n=args.nn_check, nn_k=args.nn_k, seed=args.seed)
        print("\n[Ambiguity NN Check] (spec-neighbors structure dispersion)")
        print(f"  check_n={nnr['nn_check_n']} | nn_k={nnr['nn_k']}")
        print(f"  mean_nn_dist           = {nnr['nn_mean_dist']:.6f}")
        print(f"  same_structure_rate    = {nnr['nn_same_structure_rate']:.3%}")
        print(f"  same_material_pattern  = {nnr['nn_same_pattern_rate']:.3%}")
        print("  (谱很近但 same_structure_rate 很低 => 多解严重)")

    # ---- Optional: teacher forcing metrics ----
    if args.do_tf:
        tfm = teacher_forcing_metrics(
            model=model,
            dev_spec=dev_spec[idxs],
            dev_struct=[dev_struct[i] for i in idxs],
            struc_word_dict=struc_word_dict,
            max_len=args.max_len,
            batch_size=args.tf_batch,
        )
        print("\n[Teacher Forcing] (next-token predictability)")
        print(f"  token_CE (NLL) = {tfm['tf_token_ce']:.4f}")
        print(f"  token_top1_acc = {tfm['tf_token_acc']:.3%}")
        print(f"  tokens_count   = {tfm['tf_tokens']}")

    # ---- Decode: greedy/sample/best-of-N ----
    oracle_mae_list = []
    greedy_mae_list = []
    sample_best_mae_list = []

    # stats
    pred_pair_cnt = Counter()
    pred_pattern_cnt = Counter()
    pred_len_cnt = Counter()
    pred_token_cnt = Counter()

    # prepare oracle mats/thks once
    oracle_mats, oracle_thks, oracle_targets = [], [], []
    for ii in idxs:
        mats_gt, thks_gt = parse_structure_tokens(dev_struct[ii])
        oracle_mats.append(mats_gt)
        oracle_thks.append(thks_gt)
        oracle_targets.append(dev_spec[ii])

    # oracle spec (GT->tmm) to verify dataset correctness
    B = max(1, int(args.tmm_batch))
    for st in range(0, len(idxs), B):
        ed = min(len(idxs), st + B)
        spec_oracle = calc_spec_tmmfast_batch(
            oracle_mats[st:ed], oracle_thks[st:ed],
            nk_dict_torch, wl_m, theta_rad,
            pol=args.pol,
            exit_medium=args.exit_medium,
            substrate_name=args.substrate_name,
        )
        targets = np.asarray(oracle_targets[st:ed], dtype=np.float32)
        mae = np.mean(np.abs(spec_oracle - targets), axis=1)
        oracle_mae_list.extend(mae.tolist())

    # decode loop (one by one; decode cannot batch easily)
    pred_tokens_greedy = []
    pred_tokens_best = []

    for j, ii in enumerate(idxs):
        spec_target = dev_spec[ii]

        # greedy
        if args.strategy in ["greedy", "both"]:
            gtoks = greedy_decode_tokens(model, struc_index_dict, struc_word_dict, spec_target, args.max_len)
        else:
            gtoks = []

        # sampling best-of-N
        if args.strategy in ["sample", "both"]:
            best_toks = None
            best_mae = None

            # 先采样 N 个结构
            cand_tokens = []
            for _ in range(max(1, args.sample_N)):
                toks = sample_decode_tokens(
                    model, struc_index_dict, struc_word_dict, spec_target, args.max_len,
                    top_k=args.top_k, top_p=args.top_p, temperature=args.temperature
                )
                cand_tokens.append(toks)

            # 对这 N 个候选，算 spec MAE 选最优
            # （这里用小 batch 加速）
            cand_mats, cand_thks = [], []
            for toks in cand_tokens:
                m, t = parse_structure_tokens(toks)
                cand_mats.append(m)
                cand_thks.append(t)

            cand_spec = calc_spec_tmmfast_batch(
                cand_mats, cand_thks,
                nk_dict_torch, wl_m, theta_rad,
                pol=args.pol,
                exit_medium=args.exit_medium,
                substrate_name=args.substrate_name,
            )
            maes = np.mean(np.abs(cand_spec - spec_target[None, :]), axis=1)
            kbest = int(np.argmin(maes))
            best_mae = float(maes[kbest])
            best_toks = cand_tokens[kbest]

            pred_tokens_best.append(best_toks)
            sample_best_mae_list.append(best_mae)

        pred_tokens_greedy.append(gtoks)

    # Now compute greedy MAE in batch (if used)
    if args.strategy in ["greedy", "both"]:
        pred_mats, pred_thks = [], []
        for toks in pred_tokens_greedy:
            m, t = parse_structure_tokens(toks)
            pred_mats.append(m)
            pred_thks.append(t)

        for st in range(0, len(idxs), B):
            ed = min(len(idxs), st + B)
            spec_pd = calc_spec_tmmfast_batch(
                pred_mats[st:ed], pred_thks[st:ed],
                nk_dict_torch, wl_m, theta_rad,
                pol=args.pol,
                exit_medium=args.exit_medium,
                substrate_name=args.substrate_name,
            )
            targets = dev_spec[np.array(idxs[st:ed])]
            mae = np.mean(np.abs(spec_pd - targets), axis=1)
            greedy_mae_list.extend(mae.tolist())

        # stats for greedy (collapse)
        for toks in pred_tokens_greedy:
            pred_pair_cnt[infer_pair_name(toks)] += 1
            pred_pattern_cnt[tokens_to_material_pattern(toks)] += 1
            pred_len_cnt[len(toks)] += 1
            for t in toks:
                pred_token_cnt[t] += 1

    # stats for best-of-N
    if args.strategy in ["sample", "both"]:
        for toks in pred_tokens_best:
            pred_pair_cnt[infer_pair_name(toks)] += 1
            pred_pattern_cnt[tokens_to_material_pattern(toks)] += 1
            pred_len_cnt[len(toks)] += 1
            for t in toks:
                pred_token_cnt[t] += 1

    # ---- print a few samples ----
    Kp = min(args.print_k, len(idxs))
    print("\n==================== SAMPLES ====================")
    for j in range(Kp):
        ii = idxs[j]
        gt_tokens = dev_struct[ii]

        print(f"\n---- sample {ii} ----")
        print(f"GT   pair: {infer_pair_name(gt_tokens)} | len={len(gt_tokens)}")
        print("GT   head:", gt_tokens[:12], "..." if len(gt_tokens) > 12 else "")
        print(f"Oracle MAE: {oracle_mae_list[j]:.6f}")

        if args.strategy in ["greedy", "both"]:
            toks = pred_tokens_greedy[j]
            print(f"PRED(greedy) pair: {infer_pair_name(toks)} | len={len(toks)}")
            print("PRED(greedy) head:", toks[:12], "..." if len(toks) > 12 else "")
            print(f"Greedy MAE: {greedy_mae_list[j]:.6f}")

        if args.strategy in ["sample", "both"]:
            toks = pred_tokens_best[j]
            print(f"PRED(best-of-{args.sample_N}) pair: {infer_pair_name(toks)} | len={len(toks)}")
            print("PRED(best) head:", toks[:12], "..." if len(toks) > 12 else "")
            print(f"Best-of-N MAE: {sample_best_mae_list[j]:.6f}")

    # ---- overall summary ----
    oracle_mae = np.asarray(oracle_mae_list, dtype=np.float32)
    print("\n==================== OVERALL ====================")
    print(f"eval samples: {len(idxs)}")
    print(f"spec_type=R_T | pol={args.pol} | exit_medium={args.exit_medium}")

    print("\n[Oracle MAE] (GT -> TMM_FAST vs dev_spec)")
    print(f"  mean  : {oracle_mae.mean():.6f}")
    print(f"  median: {np.median(oracle_mae):.6f}")
    print(f"  p90   : {np.quantile(oracle_mae, 0.90):.6f}")

    if args.strategy in ["greedy", "both"]:
        gm = np.asarray(greedy_mae_list, dtype=np.float32)
        print("\n[Greedy MAE] (Pred -> TMM_FAST vs dev_spec)")
        print(f"  mean  : {gm.mean():.6f}")
        print(f"  median: {np.median(gm):.6f}")
        print(f"  p90   : {np.quantile(gm, 0.90):.6f}")

    if args.strategy in ["sample", "both"]:
        bm = np.asarray(sample_best_mae_list, dtype=np.float32)
        print(f"\n[Best-of-{args.sample_N} MAE] (Sampling -> TMM_FAST vs dev_spec)")
        print(f"  mean  : {bm.mean():.6f}")
        print(f"  median: {np.median(bm):.6f}")
        print(f"  p90   : {np.quantile(bm, 0.90):.6f}")

        if args.strategy == "both":
            # improvement ratio
            gm = np.asarray(greedy_mae_list, dtype=np.float32)
            improve = (gm - bm)
            print("\n[Best-of-N Improvement] (Greedy - Best)")
            print(f"  mean_improve: {float(improve.mean()):.6f}")
            print(f"  p90_improve : {float(np.quantile(improve, 0.90)):.6f}")
            print("  (若 best-of-N 明显更好 => 多解存在且 greedy 选错模式/坍塌)")

    # ---- collapse stats ----
    print("\n[Pred pair distribution] (top %d)" % args.pair_topk)
    total = sum(pred_pair_cnt.values())
    for k, v in pred_pair_cnt.most_common(args.pair_topk):
        print(f"  {k:22s} {v:6d} ({v/max(total,1):.3%})")

    print("\n[Pred material-pattern distribution] (top %d)" % args.pair_topk)
    total2 = sum(pred_pattern_cnt.values())
    for k, v in pred_pattern_cnt.most_common(args.pair_topk):
        # pattern 可能很长，截断显示
        kk = k if len(k) <= 60 else (k[:57] + "...")
        print(f"  {kk:60s} {v:6d} ({v/max(total2,1):.3%})")

    print("\n[Pred length distribution] (top %d)" % args.pair_topk)
    total3 = sum(pred_len_cnt.values())
    for k, v in pred_len_cnt.most_common(args.pair_topk):
        print(f"  len={k:3d}  {v:6d} ({v/max(total3,1):.3%})")

    print("\n[Top predicted tokens] (top %d)" % args.pair_topk)
    total4 = sum(pred_token_cnt.values())
    for t, c in pred_token_cnt.most_common(args.pair_topk):
        print(f"  {t:18s} {c:7d} ({c/max(total4,1):.3%})")

    print("==================================================")
    print("Done.")


if __name__ == "__main__":
    main()
