#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_gen_fp_maxlen22_stable.py

FP 小长度可学习版（对齐你 DBR 生成器风格）：
- 只生成：
  (1) MIM: metal + cavity + metal （3层）
  (2) Hybrid: (H L)^pairs + cavity + metal （pairs<=4 -> 最多 10层）
- token = "Material_ThicknessNm"
- Spectrum = [R..., T...]，波长网格固定 0.9~1.7 um step=0.005（spec_dim 固定）

输出同 DBR：
OUT_DIR/
  Structure_train.pkl
  Spectrum_train.pkl
  Structure_dev.pkl
  Spectrum_dev.pkl
  meta_train.pkl
  meta_dev.pkl
"""

import os
import random
import pickle as pkl
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d

import torch
import tmm_fast

# =========================
# 0) 配置（和 DBR 风格保持一致）
# =========================
NUM_SAMPLES = 30000

LAMBDA0 = 0.9
LAMBDA1 = 1.7
STEP_UM = 0.005
WAVELENGTHS_UM = np.linspace(LAMBDA0, LAMBDA1, int(round((LAMBDA1 - LAMBDA0) / STEP_UM)) + 1)

# 进一步减少 token：lambda0 只取 3 个
LAMBDA0_SET_UM = [1.05, 1.31, 1.55]

NK_DIR = "./dataset/data1"
OUT_DIR = "./dataset/fp_maxlen22_stable"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_RATIO = 0.8
SPLIT_SEED = 42
GLOBAL_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128 if torch.cuda.is_available() else 32

COMPLEX_DTYPE = torch.complex64
REAL_DTYPE = torch.float32

# =========================
# 1) 材料池（强收敛，短长度）
# =========================
DBR_PAIRS: List[Tuple[str, str]] = [
    ("Ta2O5", "SiO2"),
    ("Si3N4", "SiO2"),
]
METALS = ["Ag", "Al", "TiN"]

CAVITY_MATS = ["SiO2", "ITO"]

# 最多 pairs=4 -> DBR=8层 + cavity1 + metal1 = 10层（max_len=22绰绰有余）
PAIR_MIN = 2
PAIR_MAX = 4

METAL_THK_SET_NM = [20, 30, 40]
ITO_THK_SET_NM = [120, 200, 400]

FAMILY_WEIGHTS = {
    "hybrid": 0.70,
    "mim": 0.30,
}

# =========================
# 2) nk 加载（同你 DBR）
# =========================
def load_nk_torch(materials: List[str], wavelengths_um: np.ndarray) -> Dict[str, torch.Tensor]:
    nk: Dict[str, torch.Tensor] = {}
    for mat in materials:
        path = os.path.join(NK_DIR, f"{mat}.csv")
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

def qw_thk_nm(nk_mat: torch.Tensor, wavelengths_um: np.ndarray, lambda0_um: float) -> int:
    idx0 = int(np.argmin(np.abs(wavelengths_um - lambda0_um)))
    n0 = float(np.real(nk_mat[idx0].detach().cpu().numpy()))
    t_nm = lambda0_um * 1000.0 / (4.0 * n0)
    return int(round(t_nm))

# =========================
# 3) 结构生成（短长度）
# =========================
def sample_family() -> str:
    r = random.random()
    return "hybrid" if r < FAMILY_WEIGHTS["hybrid"] else "mim"

def gen_hybrid_fp(nk_dict_torch: Dict[str, torch.Tensor]) -> Tuple[List[str], List[int], dict]:
    H, L = random.choice(DBR_PAIRS)
    pairs = random.randint(PAIR_MIN, PAIR_MAX)
    lambda0_um = float(random.choice(LAMBDA0_SET_UM))

    mats, thks = [], []
    # (H L)^pairs
    for i in range(pairs * 2):
        m = H if (i % 2 == 0) else L
        mats.append(m)
        thks.append(qw_thk_nm(nk_dict_torch[m], WAVELENGTHS_UM, lambda0_um))

    # cavity
    cav = random.choice(CAVITY_MATS)
    mats.append(cav)
    if cav == "ITO":
        thks.append(int(random.choice(ITO_THK_SET_NM)))
        cav_mode = "fixed"
    else:
        # 用“2*QW”当近似 half-wave（只 1 档，减少 token）
        thks.append(int(2 * qw_thk_nm(nk_dict_torch[cav], WAVELENGTHS_UM, lambda0_um)))
        cav_mode = "2qw"

    # metal
    metal = random.choice(METALS)
    mats.append(metal)
    thks.append(int(random.choice(METAL_THK_SET_NM)))

    meta = dict(
        family="hybrid",
        mirror_pair=f"{H}/{L}",
        lambda0_um=lambda0_um,
        pairs=int(pairs),
        cavity_mat=cav,
        cavity_mode=cav_mode,
        metal=metal,
        num_layers=int(len(mats)),
    )
    return mats, thks, meta

def gen_mim_fp() -> Tuple[List[str], List[int], dict]:
    metal = random.choice(METALS)
    mats = [metal, "ITO", metal]
    thks = [int(random.choice(METAL_THK_SET_NM)), int(random.choice(ITO_THK_SET_NM)), int(random.choice(METAL_THK_SET_NM))]
    meta = dict(
        family="mim",
        metal=metal,
        cavity_mat="ITO",
        num_layers=3,
    )
    return mats, thks, meta

def generate_one(nk_dict_torch):
    fam = sample_family()
    if fam == "hybrid":
        return gen_hybrid_fp(nk_dict_torch)
    return gen_mim_fp()

# =========================
# 4) batch 打包 + tmm_fast（同 DBR）
# =========================
def pack_batch_to_tmm_inputs(
    batch_mats: List[List[str]],
    batch_thks_nm: List[List[int]],
    nk_dict_torch: Dict[str, torch.Tensor],
    wl_m: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = len(batch_mats)
    W = wl_m.shape[0]
    Lmax = max(len(x) for x in batch_mats)

    d = torch.zeros((B, Lmax + 2), dtype=REAL_DTYPE, device=DEVICE)
    d[:, 0] = float("inf")
    d[:, -1] = float("inf")

    n = torch.empty((B, Lmax + 2, W), dtype=COMPLEX_DTYPE, device=DEVICE)
    n[:, 0, :] = (1.0 + 0.0j)   # air
    n[:, -1, :] = (1.0 + 0.0j)  # air（你要 glass 就改这里）

    for bi in range(B):
        mats = batch_mats[bi]
        thks = batch_thks_nm[bi]
        L = len(mats)

        d[bi, 1:1 + L] = torch.tensor(thks, dtype=REAL_DTYPE, device=DEVICE) * 1e-9
        for li, m in enumerate(mats, start=1):
            n[bi, li, :] = nk_dict_torch[m]

        if L < Lmax:
            n_pad = n[bi, L, :].clone()
            n[bi, 1 + L:1 + Lmax, :] = n_pad

    return n, d

def calc_RT_fast_batch(batch_mats, batch_thks_nm, nk_dict_torch, wl_m, theta_rad, pol="s"):
    n, d = pack_batch_to_tmm_inputs(batch_mats, batch_thks_nm, nk_dict_torch, wl_m)
    out = tmm_fast.coh_tmm(pol, n, d, theta_rad, wl_m)
    R = out["R"]; T = out["T"]
    if R.ndim == 3:
        R = R[:, 0, :]; T = T[:, 0, :]
    return (
        R.detach().cpu().float().numpy(),
        T.detach().cpu().float().numpy(),
    )

# =========================
# 5) 保存 split（同 DBR）
# =========================
def save_split(struct_list, spec_list, meta_list, out_dir):
    N = len(struct_list)
    rng = np.random.default_rng(SPLIT_SEED)
    idx = rng.permutation(N)
    split = int(N * TRAIN_RATIO)
    tr = idx[:split]
    dv = idx[split:]

    def pick(arr, ids): return [arr[i] for i in ids]

    with open(os.path.join(out_dir, "Structure_train.pkl"), "wb") as f: pkl.dump(pick(struct_list, tr), f)
    with open(os.path.join(out_dir, "Spectrum_train.pkl"), "wb") as f: pkl.dump(pick(spec_list, tr), f)
    with open(os.path.join(out_dir, "Structure_dev.pkl"), "wb") as f: pkl.dump(pick(struct_list, dv), f)
    with open(os.path.join(out_dir, "Spectrum_dev.pkl"), "wb") as f: pkl.dump(pick(spec_list, dv), f)
    with open(os.path.join(out_dir, "meta_train.pkl"), "wb") as f: pkl.dump(pick(meta_list, tr), f)
    with open(os.path.join(out_dir, "meta_dev.pkl"), "wb") as f: pkl.dump(pick(meta_list, dv), f)

    print("\n== Saved ==")
    print("Train:", len(tr), "Dev:", len(dv))
    print("OUT_DIR =", out_dir)

# =========================
# 6) 主流程
# =========================
def main():
    print("DEVICE =", DEVICE, "| BATCH_SIZE =", BATCH_SIZE)
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_SEED)

    needed = set()
    for H, L in DBR_PAIRS:
        needed.add(H); needed.add(L)
    for m in METALS: needed.add(m)
    for m in CAVITY_MATS: needed.add(m)

    missing = [m for m in sorted(needed) if not os.path.exists(os.path.join(NK_DIR, f"{m}.csv"))]
    if missing:
        raise FileNotFoundError("Missing nk csv:\n  " + "\n  ".join(missing) + f"\nExpected under: {NK_DIR}")

    nk_dict_torch = load_nk_torch(sorted(list(needed)), WAVELENGTHS_UM)

    wl_m = torch.tensor(WAVELENGTHS_UM * 1e-6, dtype=REAL_DTYPE, device=DEVICE)
    theta_rad = torch.tensor([0.0], dtype=REAL_DTYPE, device=DEVICE)

    struct_list, spec_list, meta_list = [], [], []
    pbar = tqdm(total=NUM_SAMPLES, desc="Generating FP(maxlen22) dataset (tmm_fast batch)")

    while len(struct_list) < NUM_SAMPLES:
        cur = min(BATCH_SIZE, NUM_SAMPLES - len(struct_list))
        batch_mats, batch_thks, batch_meta = [], [], []

        for _ in range(cur):
            mats, thks, meta = generate_one(nk_dict_torch)
            batch_mats.append(mats)
            batch_thks.append(thks)
            batch_meta.append(meta)

        Rb, Tb = calc_RT_fast_batch(batch_mats, batch_thks, nk_dict_torch, wl_m, theta_rad, pol="s")

        for bi in range(cur):
            toks = [f"{m}_{int(t)}" for m, t in zip(batch_mats[bi], batch_thks[bi])]
            spec = np.concatenate([Rb[bi], Tb[bi]], axis=0).astype(np.float32).tolist()
            struct_list.append(toks)
            spec_list.append(spec)
            meta_list.append(batch_meta[bi])

        pbar.update(cur)

    pbar.close()
    save_split(struct_list, spec_list, meta_list, OUT_DIR)

    # quick stats
    lens = [len(x) for x in struct_list]
    print("\nstructure len: min/mean/max =", min(lens), sum(lens)/len(lens), max(lens))
    print("spec_dim =", len(spec_list[0]), "unique_spec_len =", len(set(len(x) for x in spec_list)))

    tok_cnt = Counter()
    for seq in struct_list:
        for t in seq: tok_cnt[t] += 1
    print("OBSERVED_UNIQUE_TOKENS_IN_DATA =", len(tok_cnt))
    print("Top tokens:", tok_cnt.most_common(10))

    print("\nDone.")

if __name__ == "__main__":
    main()
