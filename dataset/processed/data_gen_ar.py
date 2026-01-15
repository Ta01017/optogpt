#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_ar_30k_smalltoken_tmm_fast_batch.py

AR (Anti-Reflection) coating dataset generator (small-token, 30k):
- token = "Material_ThicknessNm" (discrete thickness)
- structures use common industry AR stacks:
  (A) single-layer AR: L
  (B) double-layer: HL or LH
  (C) triple-layer: LHL or HLH
  (D) 5-layer: LHLHL or HLHLH

- thickness tokens are discrete optical-thickness rules at lambda0:
  t = m * lambda0/(8n), with m in {1,2} -> (1/8, 1/4) wave
  (m=2 is quarter-wave, biased)

- Uses tmm_fast.coh_tmm with batch + wavelength vectorization

IMPORTANT FIX:
- tmm_fast may assert for lossy semi-infinite exit medium.
  For Glass_Substrate, we force k=0 by taking real(nk).

Output:
OUT_DIR/
  Structure_train.pkl
  Spectrum_train.pkl
  Structure_dev.pkl
  Spectrum_dev.pkl
  meta_train.pkl
  meta_dev.pkl

Spectrum: [R..., T...] on wavelength grid 0.9~1.7 um step=0.005
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
# 0) Config
# =========================
NUM_SAMPLES = 30000

WAVELENGTHS_UM = np.linspace(0.9, 1.7, int(round((1.7 - 0.9) / 0.005)) + 1)

# ↓↓↓ 减少 token：lambda0 从 6 -> 3（更聚焦更好学）
LAMBDA0_SET_UM = [1.05, 1.31, 1.55]

NK_DIR = "./dataset/data1"
OUT_DIR = "./dataset/ar_30k_smalltoken_gpu"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_RATIO = 0.8
SPLIT_SEED = 42
GLOBAL_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128 if torch.cuda.is_available() else 32

COMPLEX_DTYPE = torch.complex64
REAL_DTYPE = torch.float32


# =========================
# 1) Materials
# =========================
SUBSTRATE = "Glass_Substrate"  # BK5/BK7 类

# AR 用低/高折常用池（控制材料空间，减少token）
AR_LOW_POOL = ["MgF2", "SiO2"]
AR_HIGH_POOL = ["TiO2", "Ta2O5", "HfO2", "Nb2O5", "Si3N4", "AlN"]

# layer count family weights (常见分布)
FAMILY_WEIGHTS = {
    "1L": 0.25,
    "2L": 0.35,
    "3L": 0.30,
    "5L": 0.10,
}

# ↓↓↓ 减少 token：mult 从 3 -> 2（只保留 1/8 和 1/4 波）
OT_MULT_SET = [1, 2]


# =========================
# 2) Utils
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        nk_t = torch.tensor(nk_np, dtype=COMPLEX_DTYPE, device=DEVICE).to(COMPLEX_DTYPE)
        nk[mat] = nk_t
    return nk


def precompute_ar_thickness_table_nm(
    nk_dict_torch: Dict[str, torch.Tensor],
    wavelengths_um: np.ndarray,
    lambda0_set_um: List[float],
    mult_set: List[int],
) -> Dict[str, Dict[Tuple[int, int], int]]:
    """
    thk_table[mat][(lambda0_nm, mult_m)] = t_nm_int
    t_nm = m * lambda0/(8n)  with m in {1,2}
    """
    wl = wavelengths_um
    thk_table: Dict[str, Dict[Tuple[int, int], int]] = {}
    for mat, nk_arr_t in nk_dict_torch.items():
        thk_table[mat] = {}
        nk_arr = nk_arr_t.detach().cpu().numpy()
        for lam0 in lambda0_set_um:
            idx0 = int(np.argmin(np.abs(wl - lam0)))
            n0 = float(np.real(nk_arr[idx0]))
            lam0_nm = int(round(lam0 * 1000))
            for m in mult_set:
                t_nm = (m * lam0 * 1000.0) / (8.0 * n0)
                thk_table[mat][(lam0_nm, int(m))] = int(round(t_nm))
    return thk_table


def sample_family() -> str:
    keys = list(FAMILY_WEIGHTS.keys())
    probs = np.array([FAMILY_WEIGHTS[k] for k in keys], dtype=np.float64)
    probs = probs / probs.sum()
    return np.random.choice(keys, p=probs).item()


def gen_ar_structure(
    thk_table: Dict[str, Dict[Tuple[int, int], int]]
) -> Tuple[List[str], List[int], dict]:
    family = sample_family()
    lambda0_um = random.choice(LAMBDA0_SET_UM)
    lambda0_nm = int(round(lambda0_um * 1000))

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
        raise RuntimeError("Unknown family: " + family)

    thks = []
    mults = []
    for mat in seq:
        # 强偏置 quarter-wave（m=2），少量 1/8 波（m=1）
        if random.random() < 0.80:
            m = 2
        else:
            m = 1
        mults.append(m)
        thks.append(thk_table[mat][(lambda0_nm, m)])

    meta = dict(
        family="AR",
        ar_family=family,
        lambda0_um=float(lambda0_um),
        lambda0_nm=int(lambda0_nm),
        low_mat=L,
        high_mat=H,
        seq=seq,
        mults=mults,
        num_layers=int(len(seq)),
        substrate=SUBSTRATE,
        incidence="air",
        exit_medium="substrate",
    )
    return seq, thks, meta


def pack_batch_to_tmm_inputs(
    batch_mats: List[List[str]],
    batch_thks_nm: List[List[int]],
    nk_dict_torch: Dict[str, torch.Tensor],
    wl_m: torch.Tensor,
    use_substrate_as_exit: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    n: [B, Lmax+2, W] complex
    d: [B, Lmax+2] real meters, ends inf
    Default: air -> films -> substrate

    IMPORTANT FIX:
      tmm_fast 对 “有损半无限出射介质” 很敏感，这里对 substrate 强制 k=0：
      n_out = real(nk_substrate)
    """
    B = len(batch_mats)
    W = wl_m.shape[0]
    Lmax = max(len(x) for x in batch_mats)

    d = torch.zeros((B, Lmax + 2), dtype=REAL_DTYPE, device=DEVICE)
    d[:, 0] = float("inf")
    d[:, -1] = float("inf")

    n = torch.empty((B, Lmax + 2, W), dtype=COMPLEX_DTYPE, device=DEVICE)
    n[:, 0, :] = (1.0 + 0.0j)

    if use_substrate_as_exit:
        n_out_real = torch.real(nk_dict_torch[SUBSTRATE]).to(REAL_DTYPE)
        n[:, -1, :] = n_out_real.to(COMPLEX_DTYPE) + 0.0j
    else:
        n[:, -1, :] = (1.0 + 0.0j)

    for bi in range(B):
        mats = batch_mats[bi]
        thks = batch_thks_nm[bi]
        L = len(mats)

        d[bi, 1:1 + L] = torch.tensor(thks, dtype=REAL_DTYPE, device=DEVICE) * 1e-9
        for li, m in enumerate(mats, start=1):
            n[bi, li, :] = nk_dict_torch[m]

        if L < Lmax:
            # pad: thickness=0, n copy last real layer
            n_pad_val = n[bi, L, :].clone()
            n[bi, 1 + L:1 + Lmax, :] = n_pad_val

    return n, d


@torch.no_grad()
def calc_RT_fast_batch(
    batch_mats: List[List[str]],
    batch_thks_nm: List[List[int]],
    nk_dict_torch: Dict[str, torch.Tensor],
    wl_m: torch.Tensor,
    theta_rad: torch.Tensor,
    pol: str = "s",
    use_substrate_as_exit: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    n, d = pack_batch_to_tmm_inputs(batch_mats, batch_thks_nm, nk_dict_torch, wl_m, use_substrate_as_exit)
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


def save_split(struct_list, spec_list, meta_list, out_dir):
    N = len(struct_list)
    rng = np.random.default_rng(SPLIT_SEED)
    idx = rng.permutation(N)
    split = int(N * TRAIN_RATIO)
    idx_train = idx[:split]
    idx_dev = idx[split:]

    train_struct = [struct_list[i] for i in idx_train]
    train_spec   = [spec_list[i] for i in idx_train]
    train_meta   = [meta_list[i] for i in idx_train]

    dev_struct = [struct_list[i] for i in idx_dev]
    dev_spec   = [spec_list[i] for i in idx_dev]
    dev_meta   = [meta_list[i] for i in idx_dev]

    with open(os.path.join(out_dir, "Structure_train.pkl"), "wb") as f:
        pkl.dump(train_struct, f)
    with open(os.path.join(out_dir, "Spectrum_train.pkl"), "wb") as f:
        pkl.dump(train_spec, f)
    with open(os.path.join(out_dir, "Structure_dev.pkl"), "wb") as f:
        pkl.dump(dev_struct, f)
    with open(os.path.join(out_dir, "Spectrum_dev.pkl"), "wb") as f:
        pkl.dump(dev_spec, f)

    with open(os.path.join(out_dir, "meta_train.pkl"), "wb") as f:
        pkl.dump(train_meta, f)
    with open(os.path.join(out_dir, "meta_dev.pkl"), "wb") as f:
        pkl.dump(dev_meta, f)

    print("\n== Saved ==")
    print("Train:", len(train_struct), "Dev:", len(dev_struct))
    print("OUT_DIR =", out_dir)


# =========================
# 3) Main
# =========================
def main():
    print("DEVICE =", DEVICE, "| BATCH_SIZE =", BATCH_SIZE)
    set_seed(GLOBAL_SEED)

    needed = set([SUBSTRATE])
    for m in AR_LOW_POOL + AR_HIGH_POOL:
        needed.add(m)

    missing = [m for m in sorted(needed) if not os.path.exists(os.path.join(NK_DIR, f"{m}.csv"))]
    if missing:
        raise FileNotFoundError("Missing nk csv:\n  " + "\n  ".join(missing) + f"\nExpected under: {NK_DIR}")

    wl_m = torch.tensor(WAVELENGTHS_UM * 1e-6, dtype=REAL_DTYPE, device=DEVICE)
    theta_rad = torch.tensor([0.0], dtype=REAL_DTYPE, device=DEVICE)

    nk_dict_torch = load_nk_torch(sorted(list(needed)), WAVELENGTHS_UM)
    thk_table = precompute_ar_thickness_table_nm(nk_dict_torch, WAVELENGTHS_UM, LAMBDA0_SET_UM, OT_MULT_SET)

    struct_list, spec_list, meta_list = [], [], []
    pbar = tqdm(total=NUM_SAMPLES, desc="Generating AR 30k small-token (tmm_fast batch)")

    while len(struct_list) < NUM_SAMPLES:
        cur = min(BATCH_SIZE, NUM_SAMPLES - len(struct_list))
        batch_mats, batch_thks, batch_meta = [], [], []

        for _ in range(cur):
            mats, thks, meta = gen_ar_structure(thk_table)
            batch_mats.append(mats)
            batch_thks.append(thks)
            batch_meta.append(meta)

        R_batch, T_batch = calc_RT_fast_batch(
            batch_mats=batch_mats,
            batch_thks_nm=batch_thks,
            nk_dict_torch=nk_dict_torch,
            wl_m=wl_m,
            theta_rad=theta_rad,
            pol="s",
            use_substrate_as_exit=True,
        )

        for bi in range(cur):
            mats = batch_mats[bi]
            thks = batch_thks[bi]
            meta = batch_meta[bi]

            struct_tokens = [f"{m}_{int(t)}" for m, t in zip(mats, thks)]
            spec_vec = np.concatenate([R_batch[bi], T_batch[bi]], axis=0).astype(np.float32).tolist()

            struct_list.append(struct_tokens)
            spec_list.append(spec_vec)
            meta_list.append(meta)

        pbar.update(cur)

    pbar.close()
    save_split(struct_list, spec_list, meta_list, OUT_DIR)

    # ===== stats =====
    tok_cnt = Counter()
    fam_cnt = Counter()
    lens = []

    for seq, meta in zip(struct_list, meta_list):
        fam_cnt[meta["ar_family"]] += 1
        lens.append(meta["num_layers"])
        for tok in seq:
            tok_cnt[tok] += 1

    print("\n== AR family distribution ==")
    total = sum(fam_cnt.values())
    for k, v in fam_cnt.most_common():
        print(f"  {k:4s} {v:6d} ({v/total:.3%})")

    print("\n== Structure length statistics (physical layers) ==")
    print(f"  min  = {min(lens)}")
    print(f"  mean = {float(np.mean(lens)):.3f}")
    print(f"  max  = {max(lens)}")
    print("\n== Suggested transformer max_len (with BOS/EOS) ==")
    print(f"  min required = {max(lens) + 2}")
    print(f"  recommended  = {max(lens) + 5}  (AR建议用 9~10)")

    print("\n== Top tokens (raw) ==")
    tot_tok = sum(tok_cnt.values())
    for t, c in tok_cnt.most_common(20):
        print(f"  {t:18s} {c:7d} ({c/tot_tok:.3%})")

    # theoretical token space
    possible_thk = defaultdict(set)
    for mat in AR_LOW_POOL + AR_HIGH_POOL:
        for (lam0_nm, mult_m), t in thk_table[mat].items():
            possible_thk[mat].add(int(t))
    total_vocab = sum(len(v) for v in possible_thk.values())

    print("\n== Token space summary (theoretical by design) ==")
    for mat in sorted(possible_thk.keys()):
        thks = sorted(possible_thk[mat])
        print(f"  {mat:10s} unique_thk={len(thks):3d}  thk_nm={thks}")

    print("\n== TOTAL token categories (unique Material_ThicknessNm) ==")
    print("TOTAL_VOCAB =", total_vocab)
    print("OBSERVED_UNIQUE_TOKENS_IN_DATA =", len(tok_cnt))

    print("\nDone. OUT_DIR =", OUT_DIR)


if __name__ == "__main__":
    main()
