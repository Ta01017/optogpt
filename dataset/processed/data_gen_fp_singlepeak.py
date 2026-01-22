#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_gen_fp_singlepeak.py

Strict single-peak FP dataset generator (Route-A: DBR + cavity + DBR)
--------------------------------------------------------------------
目标：生成“单峰带通”透射谱 T(λ)，在 0.9~1.7 um 内只有一个主峰（显著峰）。

结构强约束：
- DBR pair: Ta2O5 / SiO2（固定）
- Left mirror: (H L)^PAIRS
- Cavity: SiO2 half-wave (≈ 2*QW) with small detune
- Right mirror: (L H)^PAIRS（对称）
- incidence: near-normal (theta=1e-3 rad)
- boundary: air -> stack -> Glass_Substrate(if exists else air)

主要功能：
- tmm_fast batch GPU + server-safe patch
- nk 插值 endpoint-hold（禁止 extrapolate，减少边界尖刺）
- 在线验证 + 拒绝采样：多峰/平坦/边界尖刺/能量异常等
- 输出 overlay（80条） + 单独曲线（随机+代表）+ bad_examples
- 输出 pkl（结构 tokens, 光谱 [R...,T...], meta）+ summary.txt + verify_report.txt

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
  single_examples/*.png(+txt)
  bad_examples/*.png(+txt)
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
from scipy.signal import find_peaks

import torch
import tmm_fast
import matplotlib.pyplot as plt


# ==========================================================
# 0) tmm_fast stability patch (server-safe, no "-bool")
#    - Fixes:
#      (1) AssertionError: incoming vs outgoing ambiguous
#      (2) "negation the - operator on a bool tensor is not supported"
# ==========================================================
# ENABLE_TMM_FAST_PATCH = True
# if ENABLE_TMM_FAST_PATCH:
#     import tmm_fast.vectorized_tmm_dispersive_multistack as vt

#     def is_not_forward_angle_safe(n: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
#         """
#         Safe forward-angle selector:
#         Choose branch with Im(n*cos(theta)) >= 0 (decaying wave convention).
#         Return bool mask: True means "NOT forward" (i.e., should flip).
#         IMPORTANT: no asserts, no '-bool'.
#         """
#         ncostheta = n * torch.cos(angle)
#         forward = ncostheta.imag >= 0
#         return torch.logical_not(forward)

#     vt.is_not_forward_angle = is_not_forward_angle_safe


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

# 单峰 FP 的中心波长候选（建议落在 DBR stopband “中部”，曲线更干净）
LAMBDA0_SET_UM = [1.15, 1.25, 1.31, 1.40, 1.50]

NK_DIR = "./dataset/data"
OUT_DIR = "./dataset/fp_singlepeak"
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

# dtypes
REAL_DTYPE = torch.float32
COMPLEX_DTYPE = torch.complex64


# =========================
# 2) Structure constraints (single-peak FP)
# =========================
DBR_PAIR: Tuple[str, str] = ("Ta2O5", "SiO2")  # fixed
PAIRS = 4                                     # fixed symmetric mirrors
CAVITY_MAT = "SiO2"                           # fixed cavity material

# half-wave cavity = 2*QW * (1+detune), detune small to move peak slightly
CAVITY_DETUNE_SET = [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03]


# =========================
# 3) Validation targets (single peak)
# =========================
VAL_EPS = 0.02          # energy tolerance
EDGE_MARGIN_PTS = 6     # avoid peak too close to boundary

# peak constraints
PEAK_HEIGHT_MIN = 0.20
PEAK_PROMINENCE_MIN = 0.15
TMAX_MIN = 0.75

# sidelobe constraints: outside main-peak window, max(T) <= SLL_MAX
MAIN_PEAK_HALF_WINDOW_UM = 0.06  # protect band around peak center
SLL_MAX = 0.15

# FWHM constraints (um): 0.03~0.08 (30~80nm), avoid too wide / too narrow
FWHM_MIN_UM = 0.03
FWHM_MAX_UM = 0.08

# flat & spike checks (extra safety)
MIN_T_PTP = 0.01
END_SPIKE_K = 6
END_SPIKE_THRESH = 0.15
START_SPIKE_THRESH = 0.15

# rejection loops
MAX_TRIES_MULTIPLIER = 40


# =========================
# 4) Plots / samples saving
# =========================
OVERLAY_N = 80

SAVE_SINGLE_CURVES = True
SINGLE_DIR = os.path.join(OUT_DIR, "single_examples")
os.makedirs(SINGLE_DIR, exist_ok=True)
SINGLE_RANDOM_N = 20
SINGLE_REPR_N = 12

BAD_DIR = os.path.join(OUT_DIR, "bad_examples")
os.makedirs(BAD_DIR, exist_ok=True)
BAD_SAVE_MAX = 40


# ==========================================================
# 5) nk loading (endpoint hold) to avoid 1.7um spikes
# ==========================================================
def load_one_nk_csv(path: str, wavelengths_um: np.ndarray) -> np.ndarray:
    df = pd.read_csv(path).dropna()
    wl = df["wl"].values.astype(np.float64)
    n = df["n"].values.astype(np.float64)
    k = df["k"].values.astype(np.float64)

    # physical clamp
    k = np.clip(k, 0.0, None)

    # endpoint-hold (NO extrapolate)
    n_interp = interp1d(wl, n, bounds_error=False, fill_value=(n[0], n[-1]))
    k_interp = interp1d(wl, k, bounds_error=False, fill_value=(k[0], k[-1]))

    n_i = n_interp(wavelengths_um)
    k_i = k_interp(wavelengths_um)
    k_i = np.clip(k_i, 0.0, None)

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


def um_to_idx(wl_um: np.ndarray, x_um: float) -> int:
    return int(np.argmin(np.abs(wl_um - x_um)))

# ==========================================================
# 6) Structure generator: (H L)^PAIRS + cavity + (L H)^PAIRS
# ==========================================================
def gen_singlepeak_fp(nk_dict: Dict[str, torch.Tensor]) -> Tuple[List[str], List[int], dict]:
    H, L = DBR_PAIR
    lambda0_um = float(random.choice(LAMBDA0_SET_UM))

    mats: List[str] = []
    thks: List[int] = []

    # Left mirror: (H L)^PAIRS
    for i in range(PAIRS * 2):
        m = H if (i % 2 == 0) else L
        mats.append(m)
        thks.append(qw_thk_nm(nk_dict[m], WAVELENGTHS_UM, lambda0_um))

    # Cavity: half-wave of SiO2 (≈ 2*QW) with small detune
    detune = float(random.choice(CAVITY_DETUNE_SET))
    cav_qw = qw_thk_nm(nk_dict[CAVITY_MAT], WAVELENGTHS_UM, lambda0_um)
    cav_thk = int(round((2.0 * cav_qw) * (1.0 + detune)))
    cav_thk = max(1, cav_thk)

    mats.append(CAVITY_MAT)
    thks.append(cav_thk)

    # Right mirror: (L H)^PAIRS (symmetric)
    for i in range(PAIRS * 2):
        m = L if (i % 2 == 0) else H
        mats.append(m)
        thks.append(qw_thk_nm(nk_dict[m], WAVELENGTHS_UM, lambda0_um))

    meta = dict(
        family="fp_singlepeak",
        mirror_pair=f"{H}/{L}",
        pairs=PAIRS,
        lambda0_um=lambda0_um,
        cavity_mat=CAVITY_MAT,
        cavity_detune=detune,
        cavity_thk_nm=int(cav_thk),
        num_layers=int(len(mats)),
        inc_medium=INC_MEDIUM,
        exit_medium=EXIT_MEDIUM,
        theta_rad=float(THETA_RAD),
    )
    return mats, thks, meta


# ==========================================================
# 7) Pack batch -> tmm_fast inputs
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
# 8) Single-peak validation utilities
# ==========================================================
def compute_fwhm(wl_um: np.ndarray, T: np.ndarray, peak_idx: int) -> float:
    """
    估计主峰 FWHM（单位 um）。如果无法找到左右半高点，返回 np.nan
    """
    tmax = float(T[peak_idx])
    if tmax <= 0:
        return np.nan
    half = 0.5 * tmax

    # left crossing
    i = peak_idx
    while i > 0 and T[i] >= half:
        i -= 1
    if i == 0 and T[i] >= half:
        return np.nan
    left = wl_um[i] if T[i] < half else wl_um[i+1]

    # right crossing
    j = peak_idx
    while j < len(T) - 1 and T[j] >= half:
        j += 1
    if j == len(T) - 1 and T[j] >= half:
        return np.nan
    right = wl_um[j] if T[j] < half else wl_um[j-1]

    return float(right - left)


def validate_singlepeak_fp(R: np.ndarray, T: np.ndarray) -> Tuple[bool, str, dict]:
    """
    单峰 FP 验证：
    - 数值合法、能量守恒
    - T 有且只有一个“显著峰”(height+prominence)
    - 主峰不靠边
    - T_max >= TMAX_MIN
    - FWHM 在 [FWHM_MIN_UM, FWHM_MAX_UM]
    - 主峰外旁瓣 max <= SLL_MAX（排除主峰邻域）
    - 额外：平坦、边界尖刺
    """
    extra = {}

    if (not np.isfinite(R).all()) or (not np.isfinite(T).all()):
        return False, "non_finite", extra

    if (R.min() < -0.05) or (T.min() < -0.05):
        return False, "negative", extra
    if (R.max() > 1.05) or (T.max() > 1.05):
        return False, "gt1", extra

    S = R + T
    if S.max() > 1.0 + VAL_EPS:
        return False, "energy_gt1", extra
    if S.min() < -VAL_EPS:
        return False, "energy_negative", extra

    # flat
    if (T.max() - T.min()) < MIN_T_PTP:
        return False, "flat_T", extra

    # edge spikes (extra)
    k = END_SPIKE_K
    if len(T) > k:
        if (T[-1] - T[-k]) > END_SPIKE_THRESH:
            return False, "end_spike_T", extra
        if (T[k] - T[0]) > START_SPIKE_THRESH:
            return False, "start_spike_T", extra

    # peak detect
    peaks, props = find_peaks(
        T,
        height=PEAK_HEIGHT_MIN,
        prominence=PEAK_PROMINENCE_MIN,
    )

    if len(peaks) != 1:
        extra["num_peaks"] = int(len(peaks))
        return False, "multi_peak" if len(peaks) > 1 else "no_peak", extra

    p = int(peaks[0])
    tmax = float(T[p])
    extra["tmax"] = tmax
    extra["peak_idx"] = p
    extra["peak_wl_um"] = float(WAVELENGTHS_UM[p])

    if tmax < TMAX_MIN:
        return False, "tmax_low", extra

    # not near boundary
    if p < EDGE_MARGIN_PTS or p > (len(T) - 1 - EDGE_MARGIN_PTS):
        return False, "peak_near_edge", extra

    # fwhm
    fwhm = compute_fwhm(WAVELENGTHS_UM, T, p)
    extra["fwhm_um"] = float(fwhm) if np.isfinite(fwhm) else float("nan")
    if (not np.isfinite(fwhm)) or (fwhm < FWHM_MIN_UM) or (fwhm > FWHM_MAX_UM):
        return False, "fwhm_out", extra

    # sidelobe check
    wl0 = float(WAVELENGTHS_UM[p])
    mask_main = (WAVELENGTHS_UM >= (wl0 - MAIN_PEAK_HALF_WINDOW_UM)) & (WAVELENGTHS_UM <= (wl0 + MAIN_PEAK_HALF_WINDOW_UM))
    T_out = T[~mask_main]
    if T_out.size > 0:
        sll = float(np.max(T_out))
    else:
        sll = 0.0
    extra["sll_max"] = sll
    if sll > SLL_MAX:
        return False, "sll_high", extra

    return True, "ok", extra


# ==========================================================
# 9) Plot & save helpers
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


def write_lines(path: str, lines: List[str]):
    with open(path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln.rstrip("\n") + "\n")


def save_example(prefix: str, out_dir: str, tokens: List[str], meta: dict, R: np.ndarray, T: np.ndarray):
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f"{prefix}.png")
    txt = os.path.join(out_dir, f"{prefix}.txt")
    plot_single(WAVELENGTHS_UM, R, T, png, title=prefix)
    with open(txt, "w", encoding="utf-8") as f:
        f.write("tokens:\n")
        f.write(" ".join(tokens) + "\n\n")
        f.write("meta:\n")
        f.write(str(meta) + "\n")


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


def post_verify(struct_list, spec_list, meta_list, out_dir):
    """
    事后验证：再跑一次单峰检测，统计错误原因（理论上应极少）
    """
    W = len(WAVELENGTHS_UM)
    bad = Counter()

    for i in range(len(spec_list)):
        arr = np.asarray(spec_list[i], dtype=np.float32)
        R = arr[:W]
        T = arr[W:]
        ok, reason, _ = validate_singlepeak_fp(R, T)
        if not ok:
            bad[reason] += 1

    lines = []
    lines.append("== POST VERIFY REPORT ==")
    lines.append(f"total = {len(spec_list)}")
    lines.append("bad counts:")
    for k, v in bad.most_common():
        lines.append(f"  {k:16s}: {v}")

    path = os.path.join(out_dir, "verify_report.txt")
    write_lines(path, lines)
    print(f"[Verify] Saved: {path}")

# ==========================================================
# 10) Main
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
    H, L = DBR_PAIR
    needed.add(H); needed.add(L)
    needed.add(CAVITY_MAT)

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

    struct_list: List[List[str]] = []
    spec_list: List[List[float]] = []
    meta_list: List[dict] = []

    overlay_R: List[np.ndarray] = []
    overlay_T: List[np.ndarray] = []

    reject_cnt = Counter()
    accept_stats = {
        "tmax": [],
        "peak_wl": [],
        "fwhm_um": [],
        "sll_max": [],
    }

    bad_saved = 0

    max_tries = NUM_SAMPLES * MAX_TRIES_MULTIPLIER
    tries = 0

    pbar = tqdm(total=NUM_SAMPLES, desc="Generating single-peak FP dataset")

    while len(struct_list) < NUM_SAMPLES and tries < max_tries:
        cur = min(BATCH_SIZE, NUM_SAMPLES - len(struct_list))
        tries += cur

        batch_mats, batch_thks, batch_meta = [], [], []
        for _ in range(cur):
            mats, thks, meta = gen_singlepeak_fp(nk_dict)
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

            ok, reason, extra = validate_singlepeak_fp(R, T)
            if not ok:
                reject_cnt[reason] += 1

                if bad_saved < BAD_SAVE_MAX:
                    toks = [f"{m}_{int(t)}" for m, t in zip(batch_mats[bi], batch_thks[bi])]
                    prefix = f"bad_{bad_saved:03d}_{reason}"
                    meta_dbg = dict(batch_meta[bi])
                    meta_dbg.update({"reject_reason": reason, **extra})
                    save_example(prefix, BAD_DIR, toks, meta_dbg, R, T)
                    bad_saved += 1
                continue

            toks = [f"{m}_{int(t)}" for m, t in zip(batch_mats[bi], batch_thks[bi])]
            spec = np.concatenate([R, T], axis=0).astype(np.float32).tolist()

            meta = dict(batch_meta[bi])
            meta.update({
                "valid": True,
                "tmax": float(extra.get("tmax", float(T.max()))),
                "peak_idx": int(extra.get("peak_idx", int(np.argmax(T)))),
                "peak_wl_um": float(extra.get("peak_wl_um", float(WAVELENGTHS_UM[int(np.argmax(T))]))),
                "fwhm_um": float(extra.get("fwhm_um", float("nan"))),
                "sll_max": float(extra.get("sll_max", float("nan"))),
                "exit_used": exit_used,
            })

            struct_list.append(toks)
            spec_list.append(spec)
            meta_list.append(meta)

            accept_stats["tmax"].append(meta["tmax"])
            accept_stats["peak_wl"].append(meta["peak_wl_um"])
            accept_stats["fwhm_um"].append(meta["fwhm_um"])
            accept_stats["sll_max"].append(meta["sll_max"])

            if len(overlay_T) < OVERLAY_N:
                overlay_R.append(R.copy())
                overlay_T.append(T.copy())

            pbar.update(1)
            if len(struct_list) >= NUM_SAMPLES:
                break

    pbar.close()

    if len(struct_list) < NUM_SAMPLES:
        print(f"[Warn] Only generated {len(struct_list)}/{NUM_SAMPLES} valid samples.")
        print("You can relax constraints (TMAX_MIN, SLL_MAX, FWHM range) or increase max tries.")
    else:
        print("[OK] Generated required samples.")

    # save splits
    save_split(struct_list, spec_list, meta_list, OUT_DIR)

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

    # single-curve exports
    if SAVE_SINGLE_CURVES and len(spec_list) > 0:
        W = len(WAVELENGTHS_UM)
        rng = np.random.default_rng(0)

        # random
        rand_ids = rng.choice(len(spec_list), size=min(SINGLE_RANDOM_N, len(spec_list)), replace=False)
        for j, idx in enumerate(rand_ids):
            arr = np.asarray(spec_list[idx], dtype=np.float32)
            R = arr[:W]; T = arr[W:]
            prefix = f"random_{j:02d}_idx{idx:06d}"
            save_example(prefix, SINGLE_DIR, struct_list[idx], meta_list[idx], R, T)

        # representative: by tmax quantiles + peak at left/right
        tmaxs = np.array([m.get("tmax", 0.0) for m in meta_list], dtype=np.float32)
        order = np.argsort(tmaxs)

        pick_ids = []
        qs = [0, 5, 25, 50, 75, 95, 100]
        for q in qs:
            pos = int(round((q / 100.0) * (len(order) - 1)))
            pick_ids.append(int(order[pos]))

        peak_wls = np.array([m.get("peak_wl_um", 0.0) for m in meta_list], dtype=np.float32)
        pick_ids += [int(np.argmin(peak_wls)), int(np.argmax(peak_wls))]

        # unique + truncate
        seen = set()
        uniq = []
        for idx in pick_ids:
            if idx not in seen:
                uniq.append(idx); seen.add(idx)
        uniq = uniq[:SINGLE_REPR_N]

        for k, idx in enumerate(uniq):
            arr = np.asarray(spec_list[idx], dtype=np.float32)
            R = arr[:W]; T = arr[W:]
            prefix = f"repr_{k:02d}_tmax{tmaxs[idx]:.3f}_wl{peak_wls[idx]:.3f}_idx{idx:06d}"
            save_example(prefix, SINGLE_DIR, struct_list[idx], meta_list[idx], R, T)

        print(f"[Single] Saved random={len(rand_ids)} repr={len(uniq)} into: {SINGLE_DIR}")

    # summary
    tok_cnt = Counter()
    for seq in struct_list:
        tok_cnt.update(seq)

    lens = [len(x) for x in struct_list]
    lines = []
    lines.append("== Strict Single-Peak FP Dataset Summary ==")
    lines.append(f"OUT_DIR = {OUT_DIR}")
    lines.append(f"NUM_SAMPLES(target) = {NUM_SAMPLES}")
    lines.append(f"NUM_SAMPLES(valid)  = {len(struct_list)}")
    lines.append(f"TRIES = {tries} | MAX_TRIES = {max_tries}")
    lines.append("")
    lines.append("== Structure constraints ==")
    lines.append(f"DBR_PAIR = {DBR_PAIR} | PAIRS={PAIRS} (symmetric)")
    lines.append(f"CAVITY_MAT = {CAVITY_MAT} | cavity = half-wave (2*QW) with detune in {CAVITY_DETUNE_SET}")
    lines.append(f"LAMBDA0_SET_UM = {LAMBDA0_SET_UM}")
    lines.append("")
    lines.append("== Simulation settings ==")
    lines.append(f"INC_MEDIUM = {INC_MEDIUM}")
    lines.append(f"EXIT_MEDIUM(requested) = {EXIT_MEDIUM} | EXIT_USED = {exit_used}")
    lines.append(f"THETA_RAD = {THETA_RAD}")
    lines.append(f"WAVELENGTHS_UM = [{LAMBDA0}, {LAMBDA1}] step={STEP_UM} | W={len(WAVELENGTHS_UM)}")
    lines.append(f"ENABLE_TMM_FAST_PATCH = {ENABLE_TMM_FAST_PATCH}")
    lines.append("")
    lines.append("== Validation targets ==")
    lines.append(f"TMAX_MIN={TMAX_MIN}")
    lines.append(f"peak(height>={PEAK_HEIGHT_MIN}, prominence>={PEAK_PROMINENCE_MIN}) must be UNIQUE")
    lines.append(f"FWHM range = [{FWHM_MIN_UM}, {FWHM_MAX_UM}] um")
    lines.append(f"SLL_MAX={SLL_MAX} outside ±{MAIN_PEAK_HALF_WINDOW_UM} um around peak")
    lines.append("")
    lines.append("== Rejection reasons ==")
    total_rej = int(sum(reject_cnt.values()))
    lines.append(f"Total rejected = {total_rej}")
    for k, v in reject_cnt.most_common():
        lines.append(f"  {k:16s}: {v}")

    lines.append("")
    lines.append("== Token stats ==")
    lines.append(f"OBSERVED_UNIQUE_TOKENS_IN_DATA = {len(tok_cnt)}")
    lines.append(f"Top tokens: {tok_cnt.most_common(12)}")
    lines.append("")
    lines.append("== Structure length stats ==")
    if lens:
        lines.append(f"len(min/mean/max) = {min(lens)} / {np.mean(lens):.3f} / {max(lens)}")
    else:
        lines.append("len(min/mean/max) = 0 / 0 / 0")

    # accepted stats
    if accept_stats["tmax"]:
        tmax_arr = np.array(accept_stats["tmax"], dtype=np.float32)
        fwhm_arr = np.array(accept_stats["fwhm_um"], dtype=np.float32)
        sll_arr = np.array(accept_stats["sll_max"], dtype=np.float32)
        wl_arr = np.array(accept_stats["peak_wl"], dtype=np.float32)

        lines.append("")
        lines.append("== Accepted stats ==")
        lines.append(f"Tmax min/mean/max = {tmax_arr.min():.3f} / {tmax_arr.mean():.3f} / {tmax_arr.max():.3f}")
        lines.append(f"Peak wl min/mean/max = {wl_arr.min():.3f} / {wl_arr.mean():.3f} / {wl_arr.max():.3f} (um)")
        lines.append(f"FWHM min/mean/max = {fwhm_arr.min():.3f} / {fwhm_arr.mean():.3f} / {fwhm_arr.max():.3f} (um)")
        lines.append(f"SLL  min/mean/max = {sll_arr.min():.3f} / {sll_arr.mean():.3f} / {sll_arr.max():.3f}")

    write_lines(os.path.join(OUT_DIR, "summary.txt"), lines)
    print(f"[Summary] Saved: {os.path.join(OUT_DIR,'summary.txt')}")

    post_verify(struct_list, spec_list, meta_list, OUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
