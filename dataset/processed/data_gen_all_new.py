#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_mix_optogpt_style_v2p1_noFP_noShortAR.py

Curriculum-friendly OptoGPT-style mixed dataset generator (no FP, no short AR).

Changes vs v2:
1) Remove FP entirely.
2) AR only keeps 3L / 5L families (no 1L/2L).
3) Boundary condition unified: air -> stack -> Glass_Substrate (exit), and force substrate k=0 (real(n)).
4) Stratified split into train/dev/test with identical type ratios.
5) Token system unified: Material_ThicknessNm with thickness quantized to THK_STEP_NM grid.

Outputs:
OUT_DIR/
  Structure_train.pkl  Spectrum_train.pkl  meta_train.pkl
  Structure_dev.pkl    Spectrum_dev.pkl    meta_dev.pkl
  Structure_test.pkl   Spectrum_test.pkl   meta_test.pkl

Spectrum: [R..., T...] on wavelength grid 0.9~1.7 um step=0.005
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
# 0) Global config
# =========================================================
NUM_SAMPLES = 500000          # try 200k first; scale to 1_000_000 later
GLOBAL_SEED = 42
SPLIT_SEED = 42

# Stratified split ratios
TRAIN_RATIO = 0.80
DEV_RATIO   = 0.1
TEST_RATIO  = 0.1
assert abs(TRAIN_RATIO + DEV_RATIO + TEST_RATIO - 1.0) < 1e-9

# wavelength grid
LAMBDA0 = 0.9
LAMBDA1 = 1.7
STEP_UM = 0.005
WAVELENGTHS_UM = np.linspace(LAMBDA0, LAMBDA1, int(round((LAMBDA1 - LAMBDA0) / STEP_UM)) + 1)

# thickness quantization
THK_MIN_NM = 10
THK_MAX_NM = 500
THK_STEP_NM = 10

# Mixture (no FP)
RATIO_FIXED  = 0.80     # fixed dominates
RATIO_RANDOM = 0.20

FIXED_SUBRATIO = {
    "DBR": 0.55,   # of fixed
    "AR":  0.45,
}

# Random stacks: dielectric-only
RANDOM_MIN_LAYERS = 2
RANDOM_MAX_LAYERS = 10
RANDOM_P_SWITCH_MAT = 0.70

# Optics environment (unified)
POL = "s"
INC_MEDIUM = "air"
EXIT_MEDIUM = "substrate"
SUBSTRATE = "Glass_Substrate"
FORCE_SUBSTRATE_K0 = True

# Fixed structures
DBR_INDUSTRY_CORE: List[Tuple[str, str]] = [
    ("TiO2", "SiO2"),
    ("Ta2O5", "SiO2"),
    ("HfO2", "SiO2"),
    ("Nb2O5", "SiO2"),
    ("Si3N4", "SiO2"),
    ("AlN", "SiO2"),
]
DBR_PAIR_MIN = 6
DBR_PAIR_MAX = 10  # 12~20 layers

AR_LOW_POOL  = ["MgF2", "SiO2"]
AR_HIGH_POOL = ["TiO2", "Ta2O5", "HfO2", "Nb2O5", "Si3N4", "AlN"]
# AR only keep 3L/5L (no short sequences)
AR_FAMILY_WEIGHTS = {"3L": 0.70, "5L": 0.30}

# Random materials (dielectrics only)
RANDOM_MATERIAL_POOL = [
    "MgF2", "SiO2", "Si3N4",
    "TiO2", "Ta2O5", "HfO2", "Nb2O5", "AlN",
]

# Center wavelength options
LAMBDA0_SET_UM = [1.05, 1.31, 1.55]

# IO
NK_DIR = "./dataset/data"
OUT_DIR = "./dataset/mix_optogpt_style_v2p1_noFP_noShortAR"
os.makedirs(OUT_DIR, exist_ok=True)

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
        n  = df["n"].values.astype(np.float64)
        k  = df["k"].values.astype(np.float64)

        n_interp = interp1d(wl, n, bounds_error=False, fill_value="extrapolate")
        k_interp = interp1d(wl, k, bounds_error=False, fill_value="extrapolate")

        nk_np = n_interp(wavelengths_um) + 1j * k_interp(wavelengths_um)
        nk[mat] = torch.tensor(nk_np, dtype=COMPLEX_DTYPE, device=DEVICE)
    return nk

def qw_thk_nm(nk_mat: torch.Tensor, lambda0_um: float) -> int:
    idx0 = int(np.argmin(np.abs(WAVELENGTHS_UM - lambda0_um)))
    n0 = float(np.real(nk_mat[idx0].detach().cpu().numpy()))
    t_nm = lambda0_um * 1000.0 / (4.0 * n0)
    return quantize_thk_nm(t_nm)

def ar_ot_thk_nm(nk_mat: torch.Tensor, lambda0_um: float, m: int) -> int:
    idx0 = int(np.argmin(np.abs(WAVELENGTHS_UM - lambda0_um)))
    n0 = float(np.real(nk_mat[idx0].detach().cpu().numpy()))
    t_nm = (m * lambda0_um * 1000.0) / (8.0 * n0)
    return quantize_thk_nm(t_nm)

def choose_exit_medium_n(nk_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    nk = nk_dict[SUBSTRATE]
    if FORCE_SUBSTRATE_K0:
        return torch.real(nk).to(REAL_DTYPE).to(COMPLEX_DTYPE) + 0.0j
    return nk

def pack_batch_to_tmm_inputs(
    batch_mats: List[List[str]],
    batch_thks_nm: List[List[int]],
    nk_dict: Dict[str, torch.Tensor],
    wl_m: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = len(batch_mats)
    W = wl_m.shape[0]
    Lmax = max(len(x) for x in batch_mats)

    d = torch.zeros((B, Lmax + 2), dtype=REAL_DTYPE, device=DEVICE)
    d[:, 0] = float("inf")
    d[:, -1] = float("inf")

    n = torch.empty((B, Lmax + 2, W), dtype=COMPLEX_DTYPE, device=DEVICE)

    # incidence: air
    n[:, 0, :] = (1.0 + 0.0j)

    # exit: substrate
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


# =========================================================
# 2) Fixed structures: DBR / AR (no short AR)
# =========================================================
def gen_dbr(nk_dict: Dict[str, torch.Tensor]) -> Tuple[List[str], List[int], Dict[str, Any]]:
    H, L = random.choice(DBR_INDUSTRY_CORE)
    pairs = random.randint(DBR_PAIR_MIN, DBR_PAIR_MAX)
    lambda0_um = float(random.choice(LAMBDA0_SET_UM))

    mats, thks = [], []
    for i in range(pairs * 2):
        m = H if (i % 2 == 0) else L
        mats.append(m)
        thks.append(qw_thk_nm(nk_dict[m], lambda0_um))

    meta = dict(type="DBR", pair=f"{H}/{L}", pairs=int(pairs), lambda0_um=float(lambda0_um), num_layers=int(len(mats)))
    return mats, thks, meta

def gen_ar(nk_dict: Dict[str, torch.Tensor]) -> Tuple[List[str], List[int], Dict[str, Any]]:
    family = sample_from_weights(AR_FAMILY_WEIGHTS)  # only 3L/5L
    lambda0_um = float(random.choice(LAMBDA0_SET_UM))
    Lm = random.choice(AR_LOW_POOL)
    Hm = random.choice(AR_HIGH_POOL)

    if family == "3L":
        seq = [Lm, Hm, Lm] if random.random() < 0.5 else [Hm, Lm, Hm]
    elif family == "5L":
        seq = [Lm, Hm, Lm, Hm, Lm] if random.random() < 0.5 else [Hm, Lm, Hm, Lm, Hm]
    else:
        raise RuntimeError("Unknown AR family")

    thks, mults = [], []
    for mat in seq:
        # keep your bias (mostly 1/4-wave)
        m = 2 if random.random() < 0.80 else 1
        mults.append(int(m))
        thks.append(ar_ot_thk_nm(nk_dict[mat], lambda0_um, m))

    meta = dict(type="AR", family=family, lambda0_um=float(lambda0_um), low=Lm, high=Hm,
                mults=mults, num_layers=int(len(seq)))
    return seq, thks, meta


# =========================================================
# 3) Random stacks (dielectrics only)
# =========================================================
def gen_random_stack() -> Tuple[List[str], List[int], Dict[str, Any]]:
    L = random.randint(RANDOM_MIN_LAYERS, RANDOM_MAX_LAYERS)
    mats, thks = [], []
    cur_mat = random.choice(RANDOM_MATERIAL_POOL)
    for i in range(L):
        if i == 0:
            mat = cur_mat
        else:
            if random.random() < RANDOM_P_SWITCH_MAT:
                cur_mat = random.choice(RANDOM_MATERIAL_POOL)
            mat = cur_mat
        mats.append(mat)

        t = random.uniform(THK_MIN_NM, THK_MAX_NM)
        thks.append(quantize_thk_nm(t))

    meta = dict(type="RANDOM", num_layers=int(L), stage="dielectric_only")
    return mats, thks, meta


# =========================================================
# 4) Sampler
# =========================================================
def sample_one(nk_dict: Dict[str, torch.Tensor]) -> Tuple[List[str], List[int], Dict[str, Any]]:
    r = random.random()
    if r < RATIO_FIXED:
        which = sample_from_weights(FIXED_SUBRATIO)
        if which == "DBR":
            return gen_dbr(nk_dict)
        if which == "AR":
            return gen_ar(nk_dict)
        raise RuntimeError("Unknown fixed type")
    else:
        return gen_random_stack()


# =========================================================
# 5) Stratified split (by meta['type'])
# =========================================================
def stratified_split_indices(meta_list: List[Dict[str, Any]], seed: int):
    rng = np.random.default_rng(seed)
    type2idx = defaultdict(list)
    for i, m in enumerate(meta_list):
        type2idx[m["type"]].append(i)

    train_idx, dev_idx, test_idx = [], [], []
    for t, ids in type2idx.items():
        ids = np.array(ids, dtype=np.int64)
        rng.shuffle(ids)
        n = len(ids)

        n_train = int(round(n * TRAIN_RATIO))
        n_dev   = int(round(n * DEV_RATIO))
        n_train = min(n_train, n)
        n_dev = min(n_dev, n - n_train)
        # remainder -> test
        train_idx.extend(ids[:n_train].tolist())
        dev_idx.extend(ids[n_train:n_train + n_dev].tolist())
        test_idx.extend(ids[n_train + n_dev:].tolist())

    rng.shuffle(train_idx); rng.shuffle(dev_idx); rng.shuffle(test_idx)
    return train_idx, dev_idx, test_idx

def save_split(prefix: str, struct_list, spec_list, meta_list, ids: List[int], out_dir: str):
    def pick(arr): return [arr[i] for i in ids]
    with open(os.path.join(out_dir, f"Structure_{prefix}.pkl"), "wb") as f:
        pkl.dump(pick(struct_list), f)
    with open(os.path.join(out_dir, f"Spectrum_{prefix}.pkl"), "wb") as f:
        pkl.dump(pick(spec_list), f)
    with open(os.path.join(out_dir, f"meta_{prefix}.pkl"), "wb") as f:
        pkl.dump(pick(meta_list), f)


# =========================================================
# 6) Main
# =========================================================
def main():
    print("DEVICE =", DEVICE, "| BATCH_SIZE =", BATCH_SIZE)
    set_seed(GLOBAL_SEED)

    # Needed materials for nk (DBR + AR + RANDOM + substrate)
    needed = set(RANDOM_MATERIAL_POOL)
    for H, L in DBR_INDUSTRY_CORE:
        needed.add(H); needed.add(L)
    for m in AR_LOW_POOL + AR_HIGH_POOL:
        needed.add(m)
    needed.add(SUBSTRATE)

    missing = [m for m in sorted(needed) if not os.path.exists(os.path.join(NK_DIR, f"{m}.csv"))]
    if missing:
        raise FileNotFoundError("Missing nk csv:\n  " + "\n  ".join(missing) + f"\nExpected under: {NK_DIR}")

    nk_dict = load_nk_torch(sorted(list(needed)), WAVELENGTHS_UM)

    wl_m = torch.tensor(WAVELENGTHS_UM * 1e-6, dtype=REAL_DTYPE, device=DEVICE)
    theta_rad = torch.tensor([0.0], dtype=REAL_DTYPE, device=DEVICE)

    struct_list, spec_list, meta_list = [], [], []
    type_cnt = Counter()
    lens = []
    tok_cnt = Counter()

    pbar = tqdm(total=NUM_SAMPLES, desc="Generating MIX v2.1 (no FP, no short AR)")
    while len(struct_list) < NUM_SAMPLES:
        cur = min(BATCH_SIZE, NUM_SAMPLES - len(struct_list))
        batch_mats, batch_thks, batch_meta = [], [], []

        for _ in range(cur):
            mats, thks, meta = sample_one(nk_dict)
            batch_mats.append(mats)
            batch_thks.append(thks)
            batch_meta.append(meta)

            type_cnt[meta["type"]] += 1
            lens.append(len(mats))

        Rb, Tb = calc_RT_fast_batch(batch_mats, batch_thks, nk_dict, wl_m, theta_rad, pol=POL)

        for bi in range(cur):
            toks = [f"{m}_{int(t)}" for m, t in zip(batch_mats[bi], batch_thks[bi])]
            spec = np.concatenate([Rb[bi], Tb[bi]], axis=0).astype(np.float32).tolist()

            struct_list.append(toks)
            spec_list.append(spec)
            meta_list.append(batch_meta[bi])

            for tt in toks:
                tok_cnt[tt] += 1

        pbar.update(cur)
    pbar.close()

    # Stratified split
    tr_idx, dv_idx, te_idx = stratified_split_indices(meta_list, SPLIT_SEED)

    save_split("train", struct_list, spec_list, meta_list, tr_idx, OUT_DIR)
    save_split("dev",   struct_list, spec_list, meta_list, dv_idx, OUT_DIR)
    save_split("test",  struct_list, spec_list, meta_list, te_idx, OUT_DIR)

    # Stats
    print("\n================= STATS =================")
    print("NUM_SAMPLES =", NUM_SAMPLES)
    print("INC/EXIT =", "air ->", SUBSTRATE, f"(k0={FORCE_SUBSTRATE_K0})")
    print("thk:", f"[{THK_MIN_NM},{THK_MAX_NM}] step={THK_STEP_NM}nm")

    print("\n== Mixture type distribution ==")
    total = sum(type_cnt.values())
    for k, v in type_cnt.most_common():
        print(f"  {k:7s} {v:8d} ({v/total:.3%})")

    print("\n== Structure length statistics ==")
    print(f"  min  = {int(min(lens))}")
    print(f"  mean = {float(np.mean(lens)):.3f}")
    print(f"  max  = {int(max(lens))}")

    thickness_bins = int((THK_MAX_NM - THK_MIN_NM) // THK_STEP_NM + 1)
    mats_used = sorted(list({tok.split('_')[0] for tok in tok_cnt.keys()}))
    approx_vocab_theory = len(mats_used) * thickness_bins

    print("\n== Token space ==")
    print("num_materials  =", len(mats_used))
    print("thickness_bins =", thickness_bins)
    print("THEORY_MAX_VOCAB ~= ", approx_vocab_theory)
    print("OBSERVED_UNIQUE_TOKENS_IN_DATA =", len(tok_cnt))

    print("\n== Split sizes (stratified by type) ==")
    print("train/dev/test =", len(tr_idx), len(dv_idx), len(te_idx))

    def count_types(ids):
        c = Counter()
        for i in ids:
            c[meta_list[i]["type"]] += 1
        return c

    for name, ids in [("train", tr_idx), ("dev", dv_idx), ("test", te_idx)]:
        c = count_types(ids)
        s = sum(c.values())
        print(f"\n{name} type ratios:")
        for k, v in c.most_common():
            print(f"  {k:7s} {v:8d} ({v/max(s,1):.3%})")

    # Suggested max_len for transformer (physical layers + BOS/EOS)
    max_layers = int(max(lens))
    print("\n== Suggested transformer max_len ==")
    print(f"  max_physical_layers = {max_layers}")
    print(f"  min_required(max_layers+2) = {max_layers + 2}")
    print(f"  recommended(max_layers+4)  = {max_layers + 4}")

    print("\nDone. OUT_DIR =", OUT_DIR)


if __name__ == "__main__":
    main()
