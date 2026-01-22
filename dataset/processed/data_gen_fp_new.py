#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_gen_fp_dbrfp_validated.py

Route-A FP dataset generator (DBR + cavity + DBR) with:
- tmm_fast batch (GPU)
- numeric-stability patch for server (avoid bool-negation + forward-angle assert)
- nk interpolation WITHOUT extrapolate (endpoint hold) to avoid boundary spikes (e.g., at 1.7um)
- validation & rejection sampling (filter flat lines, edge spikes, non-finite, energy > 1, etc.)
- overlay plots for quick visual inspection (e.g., 80 curves)
- extra single-curve exports (random + representative) for analysis
- post-verify report + saved "bad" examples for debugging
- summary stats saved to txt

Outputs:
OUT_DIR/
  Structure_train.pkl
  Spectrum_train.pkl
  Structure_dev.pkl
  Spectrum_dev.pkl
  meta_train.pkl
  meta_dev.pkl
  overlay_T.png
  overlay_R.png
  bad_examples/*.png (+txt)
  verify_examples/*.png (+txt)
  single_examples/*.png (+txt)
  summary.txt
  verify_report.txt
"""

import os
import random
import pickle as pkl
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d

import torch
import tmm_fast
import matplotlib.pyplot as plt


# ==========================================================
# 0) tmm_fast stability patch (server-safe, no "-bool")
#    - Fixes:
#      (1) AssertionError: incoming vs outgoing ambiguous
#      (2) "negation the - operator on a bool tensor is not supported"
# ==========================================================
ENABLE_TMM_FAST_PATCH = True

if ENABLE_TMM_FAST_PATCH:
    import tmm_fast.vectorized_tmm_dispersive_multistack as vt

    def is_not_forward_angle_safe(n: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """
        Safe forward-angle selector:
        Choose branch with Im(n*cos(theta)) >= 0 (decaying wave convention).
        Return bool mask: True means "NOT forward" (i.e., should flip).
        IMPORTANT: no asserts, no '-bool'.
        """
        ncostheta = n * torch.cos(angle)
        forward = ncostheta.imag >= 0
        return torch.logical_not(forward)

    vt.is_not_forward_angle = is_not_forward_angle_safe


# =========================
# 1) Config
# =========================
NUM_SAMPLES = 30000

LAMBDA0 = 0.9
LAMBDA1 = 1.7
STEP_UM = 0.005
WAVELENGTHS_UM = np.linspace(
    LAMBDA0, LAMBDA1, int(round((LAMBDA1 - LAMBDA0) / STEP_UM)) + 1
)

# Increase lambda0 options to improve peak coverage
LAMBDA0_SET_UM = [0.95, 1.05, 1.15, 1.25, 1.31, 1.45, 1.55, 1.65]

NK_DIR = "./dataset/data1"
OUT_DIR = "./dataset/fp_dbrfp_validated"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_RATIO = 0.8
SPLIT_SEED = 42
GLOBAL_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128 if torch.cuda.is_available() else 32

# boundary media
INC_MEDIUM = "air"
EXIT_MEDIUM = "Glass_Substrate"  # if missing, fallback to air

# near-normal incidence (avoid branch boundary)
THETA_RAD = 1e-3  # ~0.057 degree

# dtypes (force)
REAL_DTYPE = torch.float32
COMPLEX_DTYPE = torch.complex64


# =========================
# 2) Materials (low-loss for transmission FP)
# =========================
DBR_PAIRS: List[Tuple[str, str]] = [
    ("Ta2O5", "SiO2"),
    ("Si3N4", "SiO2"),
]
CAVITY_MATS = ["SiO2", "Si3N4"]

PAIR_MIN = 2
PAIR_MAX = 4

# cavity thickness: mult * QW * (1+detune)
CAVITY_MULT_SET = [2, 4]
CAVITY_DETUNE_SET = [-0.20, -0.10, 0.0, 0.10, 0.20]

ALLOW_ASYM = True
ASYM_DELTA_SET = [-1, 0, +1]


# =========================
# 3) Online validation thresholds (filter during generation)
# =========================
VAL_EPS = 0.02  # numeric tolerance

# flat curve detect
MIN_T_PTP = 0.01
MIN_R_PTP = 0.01

# transmission constraint
REQUIRE_TMAX = True
TMAX_MIN = 0.60

# peak not too close to band edges
EDGE_MARGIN_PTS = 5

# end-point spike detect
END_SPIKE_K = 5
END_SPIKE_THRESH = 0.20
START_SPIKE_THRESH = 0.20

# limit rejection loops
MAX_TRIES_MULTIPLIER = 20  # max tries = NUM_SAMPLES * multiplier


# =========================
# 3.5) Post-verify (after generation)
# =========================
POST_VERIFY = True
VERIFY_MAX_SAVE = 80
VERIFY_DIR = os.path.join(OUT_DIR, "verify_examples")
os.makedirs(VERIFY_DIR, exist_ok=True)

VERIFY_END_SPIKE_K = 8
VERIFY_END_SPIKE_THRESH = 0.12
VERIFY_START_SPIKE_THRESH = 0.12
VERIFY_FLAT_T_PTP = 0.008


# =========================
# 3.6) Single-curve saving (for analysis)
# =========================
SAVE_SINGLE_CURVES = True
SINGLE_DIR = os.path.join(OUT_DIR, "single_examples")
os.makedirs(SINGLE_DIR, exist_ok=True)

SINGLE_RANDOM_N = 20
SINGLE_REPR_N = 12

# overlay count (you asked ~80)
OVERLAY_N = 80


# ==========================================================
# 4) nk loading (NO extrapolate: endpoint hold)
# ==========================================================
K_FLOOR = 0.0  # keep physical (0) by default; can set to 1e-8/1e-6 if desired

def load_one_nk_csv(path: str, wavelengths_um: np.ndarray) -> np.ndarray:
    df = pd.read_csv(path).dropna()
    wl = df["wl"].values.astype(np.float64)
    n = df["n"].values.astype(np.float64)
    k = df["k"].values.astype(np.float64)

    k = np.clip(k, 0.0, None)

    # endpoint-hold (NO extrapolate)
    n_interp = interp1d(wl, n, bounds_error=False, fill_value=(n[0], n[-1]))
    k_interp = interp1d(wl, k, bounds_error=False, fill_value=(k[0], k[-1]))

    n_i = n_interp(wavelengths_um)
    k_i = k_interp(wavelengths_um)

    k_i = np.clip(k_i, 0.0, None)
    if K_FLOOR > 0:
        k_i[k_i < K_FLOOR] = K_FLOOR

    return (n_i + 1j * k_i).astype(np.complex64)


def load_nk_torch(materials: List[str], wavelengths_um: np.ndarray) -> Dict[str, torch.Tensor]:
    nk: Dict[str, torch.Tensor] = {}

    # air
    nk_air = np.ones_like(wavelengths_um, dtype=np.complex64)
    nk["air"] = torch.from_numpy(nk_air).to(device=DEVICE, dtype=COMPLEX_DTYPE)

    for mat in materials:
        if mat == "air":
            continue
        path = os.path.join(NK_DIR, f"{mat}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing nk csv: {path}")
        nk_np = load_one_nk_csv(path, wavelengths_um)
        nk[mat] = torch.from_numpy(nk_np).to(device=DEVICE, dtype=COMPLEX_DTYPE)

    return nk


def qw_thk_nm(nk_mat: torch.Tensor, wavelengths_um: np.ndarray, lambda0_um: float) -> int:
    idx0 = int(np.argmin(np.abs(wavelengths_um - lambda0_um)))
    n0 = float(torch.real(nk_mat[idx0]).detach().cpu().numpy())
    t_nm = lambda0_um * 1000.0 / (4.0 * n0)
    return max(1, int(round(t_nm)))


# ==========================================================
# 5) Structure generator: (HL)^p + cavity + (LH)^p
# ==========================================================
def gen_dbr_fp(nk_dict: Dict[str, torch.Tensor]) -> Tuple[List[str], List[int], dict]:
    H, L = random.choice(DBR_PAIRS)
    lambda0_um = float(random.choice(LAMBDA0_SET_UM))

    pL = random.randint(PAIR_MIN, PAIR_MAX)
    if ALLOW_ASYM:
        delta = int(random.choice(ASYM_DELTA_SET))
        pR = int(np.clip(pL + delta, PAIR_MIN, PAIR_MAX))
    else:
        pR = pL

    cavity_mat = random.choice(CAVITY_MATS)
    mult = int(random.choice(CAVITY_MULT_SET))
    detune = float(random.choice(CAVITY_DETUNE_SET))

    mats: List[str] = []
    thks: List[int] = []

    # left: (H L)^pL
    for i in range(pL * 2):
        m = H if (i % 2 == 0) else L
        mats.append(m)
        thks.append(qw_thk_nm(nk_dict[m], WAVELENGTHS_UM, lambda0_um))

    # cavity
    cav_qw = qw_thk_nm(nk_dict[cavity_mat], WAVELENGTHS_UM, lambda0_um)
    cav_thk = int(round(mult * cav_qw * (1.0 + detune)))
    cav_thk = max(1, cav_thk)
    mats.append(cavity_mat)
    thks.append(cav_thk)

    # right: (L H)^pR
    for i in range(pR * 2):
        m = L if (i % 2 == 0) else H
        mats.append(m)
        thks.append(qw_thk_nm(nk_dict[m], WAVELENGTHS_UM, lambda0_um))

    meta = dict(
        family="dbr_fp",
        mirror_pair=f"{H}/{L}",
        lambda0_um=lambda0_um,
        pairs_left=int(pL),
        pairs_right=int(pR),
        cavity_mat=cavity_mat,
        cavity_mult=int(mult),
        cavity_detune=float(detune),
        num_layers=int(len(mats)),
        inc_medium=INC_MEDIUM,
        exit_medium=EXIT_MEDIUM,
        theta_rad=float(THETA_RAD),
    )
    return mats, thks, meta


# ==========================================================
# 6) Pack batch -> tmm_fast
# ==========================================================
def pack_batch_to_tmm_inputs(
    batch_mats: List[List[str]],
    batch_thks_nm: List[List[int]],
    nk_dict: Dict[str, torch.Tensor],
    wl_m: torch.Tensor,
    inc_medium: str,
    exit_medium: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = len(batch_mats)
    W = wl_m.shape[0]
    Lmax = max(len(x) for x in batch_mats)

    d = torch.zeros((B, Lmax + 2), dtype=REAL_DTYPE, device=DEVICE)
    d[:, 0] = float("inf")
    d[:, -1] = float("inf")

    n = torch.empty((B, Lmax + 2, W), dtype=COMPLEX_DTYPE, device=DEVICE)

    # incident
    n[:, 0, :] = nk_dict[inc_medium]

    # exit
    if exit_medium in nk_dict:
        n[:, -1, :] = nk_dict[exit_medium]
    else:
        n[:, -1, :] = nk_dict["air"]

    for bi in range(B):
        mats = batch_mats[bi]
        thks = batch_thks_nm[bi]
        L = len(mats)

        d[bi, 1:1 + L] = torch.tensor(thks, dtype=REAL_DTYPE, device=DEVICE) * 1e-9
        for li, m in enumerate(mats, start=1):
            n[bi, li, :] = nk_dict[m]

        # padding: repeat last real layer nk
        if L < Lmax:
            n_pad = n[bi, L, :].clone()
            n[bi, 1 + L:1 + Lmax, :] = n_pad

    n = n.to(dtype=COMPLEX_DTYPE)
    d = d.to(dtype=REAL_DTYPE)
    return n, d


def calc_RT_fast_batch(
    batch_mats: List[List[str]],
    batch_thks_nm: List[List[int]],
    nk_dict: Dict[str, torch.Tensor],
    wl_m: torch.Tensor,
    theta_rad: torch.Tensor,
    pol: str = "s",
    inc_medium: str = "air",
    exit_medium: str = "Glass_Substrate",
) -> Tuple[np.ndarray, np.ndarray]:
    n, d = pack_batch_to_tmm_inputs(batch_mats, batch_thks_nm, nk_dict, wl_m, inc_medium, exit_medium)

    wl_m = wl_m.to(dtype=REAL_DTYPE)
    theta_rad = theta_rad.to(dtype=REAL_DTYPE)

    out = tmm_fast.coh_tmm(pol, n, d, theta_rad, wl_m)
    R = out["R"]
    T = out["T"]
    if R.ndim == 3:
        R = R[:, 0, :]
        T = T[:, 0, :]
    return R.detach().cpu().float().numpy(), T.detach().cpu().float().numpy()


# ==========================================================
# 7) Validation / rejection (online)
# ==========================================================
def validate_curve(R: np.ndarray, T: np.ndarray) -> Tuple[bool, str, dict]:
    extra = {}

    if (not np.isfinite(R).all()) or (not np.isfinite(T).all()):
        return False, "non_finite", extra

    if (R.min() < -0.05) or (T.min() < -0.05):
        return False, "negative", extra
    if (R.max() > 1.05) or (T.max() > 1.05):
        return False, "gt1", extra

    S = R + T
    if S.max() > (1.0 + VAL_EPS):
        return False, "energy_gt1", extra
    if S.min() < (-VAL_EPS):
        return False, "energy_negative", extra

    if (T.max() - T.min()) < MIN_T_PTP:
        return False, "flat_T", extra
    if (R.max() - R.min()) < MIN_R_PTP:
        return False, "flat_R", extra

    k = END_SPIKE_K
    if len(T) > k:
        if (T[-1] - T[-k]) > END_SPIKE_THRESH:
            return False, "end_spike_T", extra
        if (T[k] - T[0]) > START_SPIKE_THRESH:
            return False, "start_spike_T", extra

    tmax = float(T.max())
    extra["tmax"] = tmax
    if REQUIRE_TMAX and (tmax < TMAX_MIN):
        return False, "tmax_low", extra

    argmax = int(np.argmax(T))
    extra["t_argmax"] = argmax
    if argmax < EDGE_MARGIN_PTS or argmax > (len(T) - 1 - EDGE_MARGIN_PTS):
        return False, "peak_near_edge", extra

    return True, "ok", extra


# ==========================================================
# 8) Plot helpers / saving
# ==========================================================
def plot_overlay(wl_um: np.ndarray, curves: List[np.ndarray], ylabel: str, title: str, save_path: str):
    plt.figure()
    for y in curves:
        plt.plot(wl_um, y)
    plt.xlabel("Wavelength (um)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_single(wl_um: np.ndarray, R: np.ndarray, T: np.ndarray, save_path: str, title: str = ""):
    plt.figure()
    plt.plot(wl_um, R, label="R")
    plt.plot(wl_um, T, label="T")
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Value")
    plt.title(title)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def write_summary(path: str, lines: List[str]):
    with open(path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln.rstrip("\n") + "\n")


def save_single_example(i: int, reason: str, tokens: List[str], meta: dict,
                        R: np.ndarray, T: np.ndarray, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{i:06d}_{reason}"
    png = os.path.join(out_dir, f"{tag}.png")
    plot_single(WAVELENGTHS_UM, R, T, png, title=f"{reason} | idx={i}")

    txt = os.path.join(out_dir, f"{tag}.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write(f"idx: {i}\n")
        f.write(f"reason: {reason}\n")
        f.write("tokens:\n")
        f.write(" ".join(tokens) + "\n\n")
        f.write("meta:\n")
        f.write(str(meta) + "\n")


# ==========================================================
# 9) Save split
# ==========================================================
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


# ==========================================================
# 10) Post-verify (after generation)
# ==========================================================
def post_verify_dataset(struct_list, spec_list, meta_list, out_dir):
    os.makedirs(VERIFY_DIR, exist_ok=True)

    bad_reasons = Counter()
    saved = 0
    lines = []

    W = len(WAVELENGTHS_UM)
    assert all(len(s) == 2 * W for s in spec_list), "spec_dim mismatch: expect [R..., T...]"

    def save_bad(i, reason, R, T):
        nonlocal saved
        if saved >= VERIFY_MAX_SAVE:
            return
        title = f"IDX={i} | {reason}"
        png = os.path.join(VERIFY_DIR, f"bad_{saved:03d}_{reason}.png")
        plot_single(WAVELENGTHS_UM, R, T, png, title=title)

        txt = os.path.join(VERIFY_DIR, f"bad_{saved:03d}_{reason}.txt")
        with open(txt, "w", encoding="utf-8") as f:
            f.write("reason: " + reason + "\n")
            f.write("tokens:\n")
            f.write(" ".join(struct_list[i]) + "\n\n")
            f.write("meta:\n")
            f.write(str(meta_list[i]) + "\n")
        saved += 1

    for i in range(len(spec_list)):
        arr = np.asarray(spec_list[i], dtype=np.float32)
        R = arr[:W]
        T = arr[W:]

        if (not np.isfinite(R).all()) or (not np.isfinite(T).all()):
            bad_reasons["non_finite"] += 1
            save_bad(i, "non_finite", R, T)
            continue

        S = R + T
        if S.max() > 1.0 + VAL_EPS:
            bad_reasons["energy_gt1"] += 1
            save_bad(i, "energy_gt1", R, T)

        if (T.max() - T.min()) < VERIFY_FLAT_T_PTP:
            bad_reasons["flat_T"] += 1
            save_bad(i, "flat_T", R, T)

        k = VERIFY_END_SPIKE_K
        if len(T) > k:
            if (T[-1] - T[-k]) > VERIFY_END_SPIKE_THRESH:
                bad_reasons["end_spike_T"] += 1
                save_bad(i, "end_spike_T", R, T)
            if (T[k] - T[0]) > VERIFY_START_SPIKE_THRESH:
                bad_reasons["start_spike_T"] += 1
                save_bad(i, "start_spike_T", R, T)

        argmax = int(np.argmax(T))
        if argmax < EDGE_MARGIN_PTS or argmax > (len(T) - 1 - EDGE_MARGIN_PTS):
            bad_reasons["peak_near_edge"] += 1
            save_bad(i, "peak_near_edge", R, T)

    lines.append("== POST VERIFY REPORT ==")
    lines.append(f"total_samples = {len(spec_list)}")
    lines.append(f"saved_examples = {saved}")
    lines.append("")
    lines.append("== bad reasons counts (not mutually exclusive) ==")
    for k, v in bad_reasons.most_common():
        lines.append(f"  {k:16s}: {v}")

    report_path = os.path.join(out_dir, "verify_report.txt")
    write_summary(report_path, lines)
    print(f"[Verify] saved report: {report_path}")
    print(f"[Verify] saved examples: {VERIFY_DIR}")


# ==========================================================
# 11) Main
# ==========================================================
def main():
    print("DEVICE =", DEVICE, "| BATCH_SIZE =", BATCH_SIZE)
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_SEED)

    # collect required materials
    needed = set(["air"])
    for H, L in DBR_PAIRS:
        needed.add(H); needed.add(L)
    for m in CAVITY_MATS:
        needed.add(m)

    # exit medium if present
    exit_csv = os.path.join(NK_DIR, f"{EXIT_MEDIUM}.csv")
    if os.path.exists(exit_csv):
        needed.add(EXIT_MEDIUM)
        exit_used = EXIT_MEDIUM
    else:
        print(f"[Warn] {EXIT_MEDIUM}.csv not found under {NK_DIR}. Fallback to air exit.")
        exit_used = "air"

    missing = [m for m in sorted(needed) if (m != "air" and not os.path.exists(os.path.join(NK_DIR, f"{m}.csv")))]
    if missing:
        raise FileNotFoundError("Missing nk csv:\n  " + "\n  ".join(missing) + f"\nExpected under: {NK_DIR}")

    nk_dict = load_nk_torch(sorted(list(needed)), WAVELENGTHS_UM)

    wl_m = torch.tensor(WAVELENGTHS_UM * 1e-6, dtype=REAL_DTYPE, device=DEVICE)
    theta_rad = torch.tensor([THETA_RAD], dtype=REAL_DTYPE, device=DEVICE)

    # outputs
    struct_list: List[List[str]] = []
    spec_list: List[List[float]] = []
    meta_list: List[dict] = []

    # overlay sampling
    overlay_R: List[np.ndarray] = []
    overlay_T: List[np.ndarray] = []

    # save some bad examples (online reject)
    bad_dir = os.path.join(OUT_DIR, "bad_examples")
    os.makedirs(bad_dir, exist_ok=True)
    BAD_SAVE_MAX = 30
    bad_saved = 0

    # validation stats
    reject_cnt = Counter()
    tmax_hist = []

    max_tries = NUM_SAMPLES * MAX_TRIES_MULTIPLIER
    tries = 0

    pbar = tqdm(total=NUM_SAMPLES, desc="Generating FP(DBR+cavity+DBR) validated dataset")

    while len(struct_list) < NUM_SAMPLES and tries < max_tries:
        cur = min(BATCH_SIZE, NUM_SAMPLES - len(struct_list))
        tries += cur

        batch_mats, batch_thks, batch_meta = [], [], []
        for _ in range(cur):
            mats, thks, meta = gen_dbr_fp(nk_dict)
            batch_mats.append(mats)
            batch_thks.append(thks)
            batch_meta.append(meta)

        Rb, Tb = calc_RT_fast_batch(
            batch_mats, batch_thks, nk_dict, wl_m, theta_rad,
            pol="s", inc_medium=INC_MEDIUM, exit_medium=exit_used
        )

        for bi in range(cur):
            R = Rb[bi].astype(np.float32)
            T = Tb[bi].astype(np.float32)

            ok, reason, extra = validate_curve(R, T)
            if not ok:
                reject_cnt[reason] += 1
                if bad_saved < BAD_SAVE_MAX:
                    title = f"BAD[{reason}]"
                    save_path = os.path.join(bad_dir, f"bad_{bad_saved:03d}_{reason}.png")
                    plot_single(WAVELENGTHS_UM, R, T, save_path, title=title)
                    txt_path = os.path.join(bad_dir, f"bad_{bad_saved:03d}_{reason}.txt")
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write("reason: " + reason + "\n")
                        f.write("tokens:\n")
                        toks = [f"{m}_{int(t)}" for m, t in zip(batch_mats[bi], batch_thks[bi])]
                        f.write(" ".join(toks) + "\n\n")
                        f.write("meta:\n")
                        f.write(str(batch_meta[bi]) + "\n")
                    bad_saved += 1
                continue

            toks = [f"{m}_{int(t)}" for m, t in zip(batch_mats[bi], batch_thks[bi])]
            spec = np.concatenate([R, T], axis=0).astype(np.float32).tolist()

            struct_list.append(toks)
            spec_list.append(spec)

            meta = dict(batch_meta[bi])
            meta.update({
                "valid": True,
                "tmax": float(extra.get("tmax", float(T.max()))),
                "t_argmax": int(extra.get("t_argmax", int(np.argmax(T)))),
                "exit_used": exit_used,
            })
            meta_list.append(meta)

            tmax_hist.append(meta["tmax"])

            if len(overlay_T) < OVERLAY_N:
                overlay_R.append(R.copy())
                overlay_T.append(T.copy())

            pbar.update(1)
            if len(struct_list) >= NUM_SAMPLES:
                break

    pbar.close()

    if len(struct_list) < NUM_SAMPLES:
        print(f"[Warn] Only generated {len(struct_list)}/{NUM_SAMPLES} valid samples.")
        print("Consider relaxing validation thresholds or expanding structure space.")
    else:
        print("[OK] Generated required samples.")

    # save splits
    save_split(struct_list, spec_list, meta_list, OUT_DIR)

    # quick stats
    lens = [len(x) for x in struct_list]
    spec_dim = len(spec_list[0]) if spec_list else 0
    uniq_spec_len = len(set(len(x) for x in spec_list))
    tok_cnt = Counter()
    for seq in struct_list:
        tok_cnt.update(seq)

    # overlay plots
    if overlay_T:
        plot_overlay(
            WAVELENGTHS_UM, overlay_T, "T",
            f"Transmission overlay (N={len(overlay_T)})",
            os.path.join(OUT_DIR, "overlay_T.png")
        )
        plot_overlay(
            WAVELENGTHS_UM, overlay_R, "R",
            f"Reflection overlay (N={len(overlay_R)})",
            os.path.join(OUT_DIR, "overlay_R.png")
        )
        print(f"[Plot] Saved overlay_T.png and overlay_R.png to {OUT_DIR}")

    # ==========================================================
    # Single-curve exports (random + representative)
    # ==========================================================
    if SAVE_SINGLE_CURVES and len(spec_list) > 0:
        W = len(WAVELENGTHS_UM)

        rng = np.random.default_rng(0)
        rand_ids = rng.choice(len(spec_list), size=min(SINGLE_RANDOM_N, len(spec_list)), replace=False)
        for j, idx in enumerate(rand_ids):
            arr = np.asarray(spec_list[idx], dtype=np.float32)
            R = arr[:W]; T = arr[W:]
            save_single_example(idx, f"random_{j:02d}", struct_list[idx], meta_list[idx], R, T, SINGLE_DIR)

        tmaxs = np.array([float(m.get("tmax", 0.0)) for m in meta_list], dtype=np.float32)
        order = np.argsort(tmaxs)

        qs = [0, 5, 25, 50, 75, 95, 100]
        pick_ids = []
        for q in qs:
            pos = int(round((q / 100.0) * (len(order) - 1)))
            pick_ids.append(int(order[pos]))

        t_argmax = np.array([int(m.get("t_argmax", 0)) for m in meta_list], dtype=np.int32)
        left_edge = int(np.argmin(t_argmax))
        right_edge = int(np.argmax(t_argmax))
        pick_ids += [left_edge, right_edge]

        seen = set()
        uniq_pick = []
        for idx in pick_ids:
            if idx not in seen:
                uniq_pick.append(idx)
                seen.add(idx)
        uniq_pick = uniq_pick[:SINGLE_REPR_N]

        for k, idx in enumerate(uniq_pick):
            arr = np.asarray(spec_list[idx], dtype=np.float32)
            R = arr[:W]; T = arr[W:]
            reason = f"repr_{k:02d}_tmax={tmaxs[idx]:.3f}_argmax={t_argmax[idx]}"
            save_single_example(idx, reason, struct_list[idx], meta_list[idx], R, T, SINGLE_DIR)

        print(f"[Single] saved random={len(rand_ids)} repr={len(uniq_pick)} into: {SINGLE_DIR}")

    # summary
    lines = []
    lines.append("== FP Route-A (DBR+cavity+DBR) dataset summary ==")
    lines.append(f"OUT_DIR = {OUT_DIR}")
    lines.append(f"NUM_SAMPLES(target) = {NUM_SAMPLES}")
    lines.append(f"NUM_SAMPLES(valid)  = {len(struct_list)}")
    lines.append(f"TRIES = {tries} | MAX_TRIES = {max_tries}")
    lines.append("")
    lines.append("== Simulation settings ==")
    lines.append(f"INC_MEDIUM = {INC_MEDIUM}")
    lines.append(f"EXIT_MEDIUM(requested) = {EXIT_MEDIUM}")
    lines.append(f"EXIT_USED = {exit_used}")
    lines.append(f"THETA_RAD = {THETA_RAD}")
    lines.append(f"WAVELENGTHS_UM = [{LAMBDA0}, {LAMBDA1}] step={STEP_UM} | W={len(WAVELENGTHS_UM)}")
    lines.append(f"ENABLE_TMM_FAST_PATCH = {ENABLE_TMM_FAST_PATCH}")
    lines.append("")
    lines.append("== Structure stats ==")
    lines.append(f"len(min/mean/max) = {min(lens) if lens else 0} / {np.mean(lens) if lens else 0:.3f} / {max(lens) if lens else 0}")
    lines.append(f"spec_dim = {spec_dim} (R+T) | unique_spec_len = {uniq_spec_len}")
    lines.append(f"OBSERVED_UNIQUE_TOKENS_IN_DATA = {len(tok_cnt)}")
    lines.append(f"Top tokens: {tok_cnt.most_common(10)}")
    lines.append("")
    lines.append("== Online validation thresholds ==")
    lines.append(f"VAL_EPS={VAL_EPS}")
    lines.append(f"MIN_T_PTP={MIN_T_PTP} | MIN_R_PTP={MIN_R_PTP}")
    lines.append(f"REQUIRE_TMAX={REQUIRE_TMAX} | TMAX_MIN={TMAX_MIN}")
    lines.append(f"EDGE_MARGIN_PTS={EDGE_MARGIN_PTS}")
    lines.append(f"END_SPIKE_K={END_SPIKE_K} | END_SPIKE_THRESH={END_SPIKE_THRESH} | START_SPIKE_THRESH={START_SPIKE_THRESH}")
    lines.append("")
    lines.append("== Rejection reasons ==")
    total_rej = sum(reject_cnt.values())
    lines.append(f"Total rejected = {total_rej}")
    for k, v in reject_cnt.most_common():
        lines.append(f"  {k:16s}: {v}")
    lines.append("")
    if tmax_hist:
        tmax_arr = np.array(tmax_hist, dtype=np.float32)
        lines.append("== Accepted T_max stats ==")
        lines.append(f"T_max min/mean/max = {tmax_arr.min():.3f} / {tmax_arr.mean():.3f} / {tmax_arr.max():.3f}")
        for q in [5, 25, 50, 75, 95]:
            lines.append(f"  p{q:02d} = {np.percentile(tmax_arr, q):.3f}")
    else:
        lines.append("== Accepted T_max stats == (no accepted samples?)")

    write_summary(os.path.join(OUT_DIR, "summary.txt"), lines)
    print(f"[Summary] Saved: {os.path.join(OUT_DIR,'summary.txt')}")

    if POST_VERIFY:
        post_verify_dataset(struct_list, spec_list, meta_list, OUT_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
