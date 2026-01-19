#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_mix_optogpt_style_tmm_fast_batch.py

统一 token 体系（OptoGPT-style）的大混合数据集生成器：
- token = "Material_ThicknessNm"（厚度离散化：thk_nm = round(thk/STEP)*STEP）
- 数据集混合组成：
  (A) 固定结构：DBR / AR / FP
  (B) 随机结构：OptoGPT-like random stack（可控层数上限、材料池、厚度范围）
- 光谱：tmm_fast.coh_tmm batch + wavelength 向量化
- 输出：Structure/Spectrum train/dev + meta

强建议：
1) 全数据集统一 token 体系（同一套厚度离散与材料池）
2) 限制随机结构最大层数（否则 max_len 必须很大，训练更难）
3) 出射介质如果是半无限 substrate 且 nk 有微小 k，tmm_fast 可能 assert
   => 对 SUBSTRATE 强制 k=0：n_out = real(nk_substrate)

Output:
OUT_DIR/
  Structure_train.pkl
  Spectrum_train.pkl
  Structure_dev.pkl
  Spectrum_dev.pkl
  meta_train.pkl
  meta_dev.pkl

Spectrum 每条为 [R..., T...]，波长网格 0.9~1.7 um step=0.005

Usage:
python generate_mix_optogpt_style_tmm_fast_batch.py
"""

import os
import random
import pickle as pkl
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d

import torch
import tmm_fast


# =========================================================
# 0) 全局配置（你只需要改这一段）
# =========================================================
NUM_SAMPLES = 200000          # 总样本数（你要“大量”就调大）
TRAIN_RATIO = 0.8
SPLIT_SEED = 42
GLOBAL_SEED = 42

# wavelength grid
LAMBDA0 = 0.9
LAMBDA1 = 1.7
STEP_UM = 0.005
WAVELENGTHS_UM = np.linspace(LAMBDA0, LAMBDA1, int(round((LAMBDA1 - LAMBDA0) / STEP_UM)) + 1)

# ---- token 离散厚度（OptoGPT-style）----
THK_MIN_NM = 10
THK_MAX_NM = 500
THK_STEP_NM = 10  # mentor问“间隔多少” -> 就是这里：10nm

# ---- 混合比例（和你前面讨论一致的推荐起步）----
RATIO_FIXED = 0.40     # 固定结构（DBR+AR+FP）总占比
RATIO_RANDOM = 0.60    # 随机结构占比（OptoGPT-like）

# 固定结构内部占比（加起来=1）
FIXED_SUBRATIO = {
    "DBR": 0.50,   # 固定结构里 DBR 占 50% => 总体 20%
    "AR":  0.25,   # => 总体 10%
    "FP":  0.25,   # => 总体 10%
}

# ---- 层数上限（决定 max_len）----
# 你想 max_len=22 => 建议 random_max_layers<=18~20
RANDOM_MAX_LAYERS = 18
RANDOM_MIN_LAYERS = 2

# 固定结构的层数范围
DBR_PAIR_MIN = 6
DBR_PAIR_MAX = 10         # 12~20 layers
AR_FAMILY_WEIGHTS = {"1L": 0.25, "2L": 0.35, "3L": 0.30, "5L": 0.10}
FP_FAMILY_WEIGHTS = {"hybrid": 0.70, "mim": 0.30}
FP_PAIR_MIN = 2
FP_PAIR_MAX = 4           # hybrid最多 10层（你之前的设定）

# ---- 光学环境 ----
POL = "s"
INC_MEDIUM = "air"
EXIT_MEDIUM = "substrate"   # "air" or "substrate"
SUBSTRATE = "Glass_Substrate"  # 你的 BK5/BK7 csv 名称
FORCE_SUBSTRATE_K0 = True      # 解决 tmm_fast 对 lossy semi-infinite substrate 的 assert

# ---- 材料池（统一！）----
# 工业常用 DBR 对
DBR_INDUSTRY_CORE: List[Tuple[str, str]] = [
    ("TiO2", "SiO2"),
    ("Ta2O5", "SiO2"),
    ("HfO2", "SiO2"),
    ("Nb2O5", "SiO2"),
    ("Si3N4", "SiO2"),
    ("AlN", "SiO2"),
]

# AR 常用池（尽量收敛，避免 token 爆炸）
AR_LOW_POOL = ["MgF2", "SiO2"]
AR_HIGH_POOL = ["TiO2", "Ta2O5", "HfO2", "Nb2O5", "Si3N4", "AlN"]

# FP（你之前短可学版本）
FP_DBR_PAIRS: List[Tuple[str, str]] = [
    ("Ta2O5", "SiO2"),
    ("Si3N4", "SiO2"),
]
FP_METALS = ["Ag", "Al", "TiN"]
FP_CAVITY_MATS = ["SiO2", "ITO"]
FP_METAL_THK_SET_NM = [20, 30, 40]     # 金属薄层通常固定几档
FP_ITO_THK_SET_NM = [120, 200, 400]    # ITO 腔厚固定几档

# 随机结构材料池（OptoGPT-like）：建议先收敛，再逐步放开
RANDOM_MATERIAL_POOL = [
    # low/mid/high
    "MgF2", "SiO2",
    "Si3N4",
    "TiO2", "Ta2O5", "HfO2", "Nb2O5", "AlN",
    # 可选：金属/吸收（不建议一开始就放太多，会多解更严重）
    "ITO", "TiN", "Ag", "Al",
]

# 随机结构“材料切换概率”：避免全是同一种材料
RANDOM_P_SWITCH_MAT = 0.70

# 数据与输出
NK_DIR = "./dataset/data"
OUT_DIR = "./dataset/mix_optogpt_style"
os.makedirs(OUT_DIR, exist_ok=True)

# 计算设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128 if torch.cuda.is_available() else 32
COMPLEX_DTYPE = torch.complex64
REAL_DTYPE = torch.float32


# =========================================================
# 1) Utils
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

def quantize_thk_nm(thk_nm: float) -> int:
    """离散厚度到 THK_STEP_NM，并裁剪到 [THK_MIN_NM, THK_MAX_NM]."""
    q = int(round(thk_nm / THK_STEP_NM) * THK_STEP_NM)
    return clamp_int(q, THK_MIN_NM, THK_MAX_NM)

def sample_from_weights(d: Dict[str, float]) -> str:
    keys = list(d.keys())
    probs = np.array([d[k] for k in keys], dtype=np.float64)
    probs = probs / probs.sum()
    return np.random.choice(keys, p=probs).item()

def load_nk_torch(materials: List[str], wavelengths_um: np.ndarray) -> Dict[str, torch.Tensor]:
    nk: Dict[str, torch.Tensor] = {}
    for mat in materials:
        path = os.path.join(NK_DIR, f"{mat}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing nk csv for material: {mat} | expected: {path}")

        df = pd.read_csv(path).dropna()
        wl = df["wl"].values.astype(np.float64)
        n = df["n"].values.astype(np.float64)
        k = df["k"].values.astype(np.float64)

        n_interp = interp1d(wl, n, bounds_error=False, fill_value="extrapolate")
        k_interp = interp1d(wl, k, bounds_error=False, fill_value="extrapolate")

        nk_np = n_interp(wavelengths_um) + 1j * k_interp(wavelengths_um)
        nk[mat] = torch.tensor(nk_np, dtype=COMPLEX_DTYPE, device=DEVICE)
    return nk

def qw_thk_nm(nk_mat: torch.Tensor, lambda0_um: float) -> int:
    """quarter-wave thickness at lambda0: t = lambda0/(4n). 返回离散后的 nm."""
    idx0 = int(np.argmin(np.abs(WAVELENGTHS_UM - lambda0_um)))
    n0 = float(np.real(nk_mat[idx0].detach().cpu().numpy()))
    t_nm = lambda0_um * 1000.0 / (4.0 * n0)
    return quantize_thk_nm(t_nm)

def ar_ot_thk_nm(nk_mat: torch.Tensor, lambda0_um: float, m: int) -> int:
    """
    AR 用：t = m * lambda0/(8n), m in {1,2} => 1/8 或 1/4 波
    返回离散后的 nm
    """
    idx0 = int(np.argmin(np.abs(WAVELENGTHS_UM - lambda0_um)))
    n0 = float(np.real(nk_mat[idx0].detach().cpu().numpy()))
    t_nm = (m * lambda0_um * 1000.0) / (8.0 * n0)
    return quantize_thk_nm(t_nm)

def choose_exit_medium_n(nk_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """返回出射介质 n(λ)，complex tensor [W]."""
    if EXIT_MEDIUM == "air":
        return torch.ones((len(WAVELENGTHS_UM),), dtype=COMPLEX_DTYPE, device=DEVICE)
    if EXIT_MEDIUM == "substrate":
        nk = nk_dict[SUBSTRATE]
        if FORCE_SUBSTRATE_K0:
            # tmm_fast 对 lossy semi-infinite 介质很敏感：强制 k=0
            return torch.real(nk).to(REAL_DTYPE).to(COMPLEX_DTYPE) + 0.0j
        return nk
    raise ValueError("EXIT_MEDIUM must be 'air' or 'substrate'")

def pack_batch_to_tmm_inputs(
    batch_mats: List[List[str]],
    batch_thks_nm: List[List[int]],
    nk_dict: Dict[str, torch.Tensor],
    wl_m: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    n: [B, Lmax+2, W] complex
    d: [B, Lmax+2] real meters, ends inf
    """
    B = len(batch_mats)
    W = wl_m.shape[0]
    Lmax = max(len(x) for x in batch_mats)

    d = torch.zeros((B, Lmax + 2), dtype=REAL_DTYPE, device=DEVICE)
    d[:, 0] = float("inf")
    d[:, -1] = float("inf")

    n = torch.empty((B, Lmax + 2, W), dtype=COMPLEX_DTYPE, device=DEVICE)

    # incidence
    if INC_MEDIUM == "air":
        n[:, 0, :] = (1.0 + 0.0j)
    else:
        raise ValueError("Only INC_MEDIUM='air' supported here (you can extend).")

    # exit
    n_out = choose_exit_medium_n(nk_dict)  # [W]
    n[:, -1, :] = n_out.unsqueeze(0).expand(B, W)

    for bi in range(B):
        mats = batch_mats[bi]
        thks = batch_thks_nm[bi]
        L = len(mats)

        d[bi, 1:1 + L] = torch.tensor(thks, dtype=REAL_DTYPE, device=DEVICE) * 1e-9
        for li, mat in enumerate(mats, start=1):
            n[bi, li, :] = nk_dict[mat]

        if L < Lmax:
            # pad：thickness=0，n复制最后一层（安全）
            n_pad = n[bi, L, :].clone()
            n[bi, 1 + L:1 + Lmax, :] = n_pad

    return n, d

@torch.no_grad()
def calc_RT_fast_batch(
    batch_mats: List[List[str]],
    batch_thks_nm: List[List[int]],
    nk_dict: Dict[str, torch.Tensor],
    wl_m: torch.Tensor,
    theta_rad: torch.Tensor,
    pol: str = "s",
) -> Tuple[np.ndarray, np.ndarray]:
    n, d = pack_batch_to_tmm_inputs(batch_mats, batch_thks_nm, nk_dict, wl_m)
    out = tmm_fast.coh_tmm(pol, n, d, theta_rad, wl_m)
    R = out["R"]
    T = out["T"]
    if R.ndim == 3:
        R = R[:, 0, :]
        T = T[:, 0, :]
    elif R.ndim != 2:
        raise RuntimeError(f"Unexpected R shape: {tuple(R.shape)}")

    return (
        R.detach().cpu().float().numpy(),
        T.detach().cpu().float().numpy(),
    )

def save_split(struct_list, spec_list, meta_list, out_dir: str):
    N = len(struct_list)
    rng = np.random.default_rng(SPLIT_SEED)
    idx = rng.permutation(N)
    split = int(N * TRAIN_RATIO)
    idx_train = idx[:split]
    idx_dev = idx[split:]

    def pick(arr, ids): return [arr[i] for i in ids]

    with open(os.path.join(out_dir, "Structure_train.pkl"), "wb") as f:
        pkl.dump(pick(struct_list, idx_train), f)
    with open(os.path.join(out_dir, "Spectrum_train.pkl"), "wb") as f:
        pkl.dump(pick(spec_list, idx_train), f)
    with open(os.path.join(out_dir, "Structure_dev.pkl"), "wb") as f:
        pkl.dump(pick(struct_list, idx_dev), f)
    with open(os.path.join(out_dir, "Spectrum_dev.pkl"), "wb") as f:
        pkl.dump(pick(spec_list, idx_dev), f)
    with open(os.path.join(out_dir, "meta_train.pkl"), "wb") as f:
        pkl.dump(pick(meta_list, idx_train), f)
    with open(os.path.join(out_dir, "meta_dev.pkl"), "wb") as f:
        pkl.dump(pick(meta_list, idx_dev), f)

    print("\n== Saved ==")
    print("Train:", len(idx_train), "Dev:", len(idx_dev))
    print("OUT_DIR =", out_dir)


# =========================================================
# 2) 固定结构生成：DBR / AR / FP（统一厚度离散）
# =========================================================
def gen_dbr(nk_dict: Dict[str, torch.Tensor]) -> Tuple[List[str], List[int], Dict[str, Any]]:
    H, L = random.choice(DBR_INDUSTRY_CORE)
    pairs = random.randint(DBR_PAIR_MIN, DBR_PAIR_MAX)
    # 用中心波长决定QW厚度（但最终仍量化到 10nm grid）
    lambda0_um = float(random.choice([1.05, 1.31, 1.55]))
    mats, thks = [], []
    for i in range(pairs * 2):
        m = H if (i % 2 == 0) else L
        mats.append(m)
        thks.append(qw_thk_nm(nk_dict[m], lambda0_um))
    meta = dict(type="DBR", pair=f"{H}/{L}", pairs=pairs, lambda0_um=lambda0_um, num_layers=len(mats))
    return mats, thks, meta

def gen_ar(nk_dict: Dict[str, torch.Tensor]) -> Tuple[List[str], List[int], Dict[str, Any]]:
    family = sample_from_weights(AR_FAMILY_WEIGHTS)
    lambda0_um = float(random.choice([1.05, 1.31, 1.55]))
    L = random.choice(AR_LOW_POOL)
    H = random.choice(AR_HIGH_POOL)

    if family == "1L":
        seq = [L]
    elif family == "2L":
        seq = [H, L] if random.random() < 0.5 else [L, H]
    elif family == "3L":
        seq = [L, H, L] if random.random() < 0.5 else [H, L, H]
    elif family == "5L":
        seq = [L, H, L, H, L] if random.random() < 0.5 else [H, L, H, L, H]
    else:
        raise RuntimeError("Unknown AR family")

    thks, mults = [], []
    for mat in seq:
        # m in {1,2}: 强偏置 quarter-wave (m=2)
        m = 2 if random.random() < 0.80 else 1
        mults.append(m)
        thks.append(ar_ot_thk_nm(nk_dict[mat], lambda0_um, m))

    meta = dict(type="AR", family=family, lambda0_um=lambda0_um, low=L, high=H, mults=mults, num_layers=len(seq))
    return seq, thks, meta

def gen_fp(nk_dict: Dict[str, torch.Tensor]) -> Tuple[List[str], List[int], Dict[str, Any]]:
    fam = "hybrid" if random.random() < FP_FAMILY_WEIGHTS["hybrid"] else "mim"
    if fam == "mim":
        metal = random.choice(FP_METALS)
        mats = [metal, "ITO", metal]
        thks = [
            quantize_thk_nm(random.choice(FP_METAL_THK_SET_NM)),
            quantize_thk_nm(random.choice(FP_ITO_THK_SET_NM)),
            quantize_thk_nm(random.choice(FP_METAL_THK_SET_NM)),
        ]
        meta = dict(type="FP", family="mim", metal=metal, num_layers=3)
        return mats, thks, meta

    # hybrid: (HL)^pairs + cavity + metal
    H, L = random.choice(FP_DBR_PAIRS)
    pairs = random.randint(FP_PAIR_MIN, FP_PAIR_MAX)
    lambda0_um = float(random.choice([1.05, 1.31, 1.55]))

    mats, thks = [], []
    for i in range(pairs * 2):
        m = H if (i % 2 == 0) else L
        mats.append(m)
        thks.append(qw_thk_nm(nk_dict[m], lambda0_um))

    cav = random.choice(FP_CAVITY_MATS)
    mats.append(cav)
    if cav == "ITO":
        thks.append(quantize_thk_nm(random.choice(FP_ITO_THK_SET_NM)))
        cav_mode = "fixed"
    else:
        # 2*QW 近似 half-wave，仍离散到 10nm grid
        thks.append(quantize_thk_nm(2 * qw_thk_nm(nk_dict[cav], lambda0_um)))
        cav_mode = "2qw"

    metal = random.choice(FP_METALS)
    mats.append(metal)
    thks.append(quantize_thk_nm(random.choice(FP_METAL_THK_SET_NM)))

    meta = dict(type="FP", family="hybrid", mirror=f"{H}/{L}", pairs=pairs, lambda0_um=lambda0_um,
                cavity=cav, cav_mode=cav_mode, metal=metal, num_layers=len(mats))
    return mats, thks, meta


# =========================================================
# 3) 随机结构生成：OptoGPT-like random stacks
# =========================================================
def gen_random_stack(nk_dict: Dict[str, torch.Tensor]) -> Tuple[List[str], List[int], Dict[str, Any]]:
    """
    OptoGPT 风格随机数据（可控）：
    - 层数随机（2~RANDOM_MAX_LAYERS）
    - 材料从 RANDOM_MATERIAL_POOL 采样，且有切换概率，避免全同
    - 厚度在 [THK_MIN_NM, THK_MAX_NM] 连续采样后再量化到 THK_STEP_NM
    """
    L = random.randint(RANDOM_MIN_LAYERS, RANDOM_MAX_LAYERS)

    mats = []
    thks = []

    cur_mat = random.choice(RANDOM_MATERIAL_POOL)
    for i in range(L):
        if i == 0:
            mat = cur_mat
        else:
            if random.random() < RANDOM_P_SWITCH_MAT:
                mat = random.choice(RANDOM_MATERIAL_POOL)
                cur_mat = mat
            else:
                mat = cur_mat
        mats.append(mat)

        # 连续厚度 -> 量化
        t = random.uniform(THK_MIN_NM, THK_MAX_NM)
        thks.append(quantize_thk_nm(t))

    meta = dict(type="RANDOM", num_layers=L, mat_pool="RANDOM_MATERIAL_POOL")
    return mats, thks, meta


# =========================================================
# 4) 混合采样器
# =========================================================
def sample_one(nk_dict: Dict[str, torch.Tensor]) -> Tuple[List[str], List[int], Dict[str, Any]]:
    r = random.random()
    if r < RATIO_FIXED:
        which = sample_from_weights(FIXED_SUBRATIO)
        if which == "DBR":
            return gen_dbr(nk_dict)
        if which == "AR":
            return gen_ar(nk_dict)
        if which == "FP":
            return gen_fp(nk_dict)
        raise RuntimeError("Unknown fixed type")
    else:
        return gen_random_stack(nk_dict)


# =========================================================
# 5) Main
# =========================================================
def main():
    print("DEVICE =", DEVICE, "| BATCH_SIZE =", BATCH_SIZE)
    set_seed(GLOBAL_SEED)

    # 需要的材料：所有池 + substrate（如果用 substrate 出射）
    needed = set(RANDOM_MATERIAL_POOL)
    for H, L in DBR_INDUSTRY_CORE:
        needed.add(H); needed.add(L)
    for m in AR_LOW_POOL + AR_HIGH_POOL:
        needed.add(m)
    for H, L in FP_DBR_PAIRS:
        needed.add(H); needed.add(L)
    for m in FP_METALS + FP_CAVITY_MATS:
        needed.add(m)
    if EXIT_MEDIUM == "substrate":
        needed.add(SUBSTRATE)

    missing = [m for m in sorted(needed) if not os.path.exists(os.path.join(NK_DIR, f"{m}.csv"))]
    if missing:
        raise FileNotFoundError("Missing nk csv:\n  " + "\n  ".join(missing) + f"\nExpected under: {NK_DIR}")

    nk_dict = load_nk_torch(sorted(list(needed)), WAVELENGTHS_UM)

    wl_m = torch.tensor(WAVELENGTHS_UM * 1e-6, dtype=REAL_DTYPE, device=DEVICE)
    theta_rad = torch.tensor([0.0], dtype=REAL_DTYPE, device=DEVICE)

    struct_list, spec_list, meta_list = [], [], []
    pbar = tqdm(total=NUM_SAMPLES, desc="Generating MIX dataset (tmm_fast batch)")

    # 统计
    max_layers_seen = 0
    type_cnt = Counter()

    while len(struct_list) < NUM_SAMPLES:
        cur = min(BATCH_SIZE, NUM_SAMPLES - len(struct_list))
        batch_mats, batch_thks, batch_meta = [], [], []

        for _ in range(cur):
            mats, thks, meta = sample_one(nk_dict)
            # 最终安全裁剪（防止逻辑失误）
            if meta["num_layers"] > RANDOM_MAX_LAYERS and meta["type"] == "RANDOM":
                mats = mats[:RANDOM_MAX_LAYERS]
                thks = thks[:RANDOM_MAX_LAYERS]
                meta["num_layers"] = len(mats)

            batch_mats.append(mats)
            batch_thks.append(thks)
            batch_meta.append(meta)

            max_layers_seen = max(max_layers_seen, meta["num_layers"])
            type_cnt[meta["type"]] += 1

        Rb, Tb = calc_RT_fast_batch(batch_mats, batch_thks, nk_dict, wl_m, theta_rad, pol=POL)

        for bi in range(cur):
            toks = [f"{m}_{int(t)}" for m, t in zip(batch_mats[bi], batch_thks[bi])]
            spec = np.concatenate([Rb[bi], Tb[bi]], axis=0).astype(np.float32).tolist()

            struct_list.append(toks)
            spec_list.append(spec)
            meta_list.append(batch_meta[bi])

        pbar.update(cur)

    pbar.close()
    save_split(struct_list, spec_list, meta_list, OUT_DIR)

    # ========== 统计输出 ==========
    tok_cnt = Counter()
    lens = []
    for seq in struct_list:
        lens.append(len(seq))
        for t in seq:
            tok_cnt[t] += 1

    print("\n================= STATS =================")
    print("NUM_SAMPLES =", NUM_SAMPLES)
    print("INC_MEDIUM =", INC_MEDIUM, "| EXIT_MEDIUM =", EXIT_MEDIUM, "| SUBSTRATE =", SUBSTRATE)
    print("thk range/step:", f"[{THK_MIN_NM},{THK_MAX_NM}] step={THK_STEP_NM}nm")

    print("\n== Mixture type distribution ==")
    total = sum(type_cnt.values())
    for k, v in type_cnt.most_common():
        print(f"  {k:7s} {v:8d} ({v/total:.3%})")

    print("\n== Structure length statistics (physical layers) ==")
    print(f"  min  = {int(min(lens))}")
    print(f"  mean = {float(np.mean(lens)):.3f}")
    print(f"  max  = {int(max(lens))}")
    print(f"  max_layers_seen = {max_layers_seen}")

    print("\n== Suggested transformer max_len (with BOS/EOS) ==")
    # 你要 max_len=22：希望 max_layers<=18~20
    print(f"  min required = {int(max(lens)) + 2}")
    print(f"  recommended  = {int(max(lens)) + 4}")

    print("\n== Spectrum dimension ==")
    print("spec_dim =", len(spec_list[0]), "| unique_spec_len =", len(set(len(x) for x in spec_list)))

    # 理论 token 上限（按材料数 * 厚度档位）
    thickness_bins = int((THK_MAX_NM - THK_MIN_NM) // THK_STEP_NM + 1)
    mats_used = sorted(list({tok.split("_")[0] for tok in tok_cnt.keys()}))
    approx_vocab_theory = len(mats_used) * thickness_bins

    print("\n== Token space summary ==")
    print("materials_used =", mats_used)
    print("num_materials  =", len(mats_used))
    print("thickness_bins =", thickness_bins, f"(from {THK_MIN_NM} to {THK_MAX_NM} step {THK_STEP_NM})")
    print("THEORETICAL_MAX_VOCAB ~= num_materials * thickness_bins =", approx_vocab_theory)

    print("\n== OBSERVED token categories in data ==")
    print("OBSERVED_UNIQUE_TOKENS_IN_DATA =", len(tok_cnt))

    print("\n== Top tokens (raw, top20) ==")
    tot_tok = sum(tok_cnt.values())
    for t, c in tok_cnt.most_common(20):
        print(f"  {t:18s} {c:9d} ({c/tot_tok:.3%})")

    print("\nDone. OUT_DIR =", OUT_DIR)


if __name__ == "__main__":
    main()
