#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_gen_fp_dbr_cavity_dbr_maxlen22.py

路线A：DBR + cavity + DBR（双介质镜 FP 透射带通）

- 结构：
    (H L)^p_left  + cavity + (L H)^p_right
  默认左右 pairs 相等（也可允许轻微不对称）

- token: "Material_ThicknessNm"
- Spectrum: [R..., T...]，波长网格固定 0.9~1.7 um step=0.005
- 输出同 DBR：
    OUT_DIR/
      Structure_train.pkl
      Spectrum_train.pkl
      Structure_dev.pkl
      Spectrum_dev.pkl
      meta_train.pkl
      meta_dev.pkl

- 额外：随机采样若干条曲线叠图（T 和 R 各一张），用于快速目检有效性
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

# ===== tmm_fast forward-angle assert workaround =====
import tmm_fast.vectorized_tmm_dispersive_multistack as vt

_orig_is_not_forward_angle = vt.is_not_forward_angle

def _patched_is_not_forward_angle(n, angle):
    """
    tmm_fast 在极小入射角 + 近乎无损介质(k~0)时，
    forward/backward 分支判定会触发 assert。

    这里做一个工程化处理：
    - 当 |angle| 很小（近似正入射）时，直接认为是 forward（返回 False）
    - 否则回退原始逻辑
    """
    # angle shape: (B, 1, W) or similar, complex dtype inside vt
    # 用实部判定大小即可
    ang = angle.real.abs()
    if torch.max(ang).item() < 1e-3:   # 约 0.057°，足够“小角度”
        return torch.zeros_like(ang, dtype=torch.bool)
    return _orig_is_not_forward_angle(n, angle)

vt.is_not_forward_angle = _patched_is_not_forward_angle
# ===== end workaround =====




# =========================
# 0) 配置
# =========================
NUM_SAMPLES = 30000

LAMBDA0 = 0.9
LAMBDA1 = 1.7
STEP_UM = 0.005
WAVELENGTHS_UM = np.linspace(LAMBDA0, LAMBDA1, int(round((LAMBDA1 - LAMBDA0) / STEP_UM)) + 1)

# 为了让峰值覆盖更广：lambda0 给更多离散点（仍然不至于 token 爆炸）
LAMBDA0_SET_UM = [0.95, 1.05, 1.15, 1.25, 1.31, 1.45, 1.55, 1.65]

NK_DIR = "./dataset/data"
OUT_DIR = "./dataset/fp_mew"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_RATIO = 0.8
SPLIT_SEED = 42
GLOBAL_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128 if torch.cuda.is_available() else 32

COMPLEX_DTYPE = torch.complex64
REAL_DTYPE = torch.float32

# 介质边界：入射 air，出射 glass（如果没有 Glass_Substrate.csv，则自动退化成 air）
INC_MEDIUM = "air"
EXIT_MEDIUM = "Glass_Substrate"  # 若不存在对应csv，会fallback为 air

# =========================
# 1) 材料与结构空间（FP 透射优先，尽量低损耗）
# =========================
# DBR 材料对（高/低折射率介质）
DBR_PAIRS: List[Tuple[str, str]] = [
    ("Ta2O5", "SiO2"),
    ("Si3N4", "SiO2"),
]

# 腔材料：尽量选低损耗介质（别用 ITO/金属当腔，否则 T 峰很难高）
CAVITY_MATS = ["SiO2", "Si3N4"]

# DBR 对数范围：pairs<=6 -> 每侧最多 12 层 + cavity 1 层 = 25 层（max_len=22 可能不够）
# 你说 max_len=22：那物理层数要 <= 10 左右更稳（22 还要加 BOS/EOS/PAD）
# 所以这里把 pairs 控制在 2~4：每侧 4~8 层，总计 9~17 物理层（对 22 很安全）
PAIR_MIN = 2
PAIR_MAX = 4

# 腔厚度策略：在“半波腔”的基础上做离散 detune，提升覆盖与多样性
# cavity_thk = m * QW * (1 + detune)
CAVITY_MULT_SET = [2, 4]                 # 2QW (≈half-wave), 4QW (≈full-wave)
CAVITY_DETUNE_SET = [-0.20, -0.10, 0.0, 0.10, 0.20]

# 左右镜是否允许轻微不对称（提高数据多样性）
ALLOW_ASYM = True
ASYM_DELTA_SET = [-1, 0, +1]  # pairs_right = pairs_left + delta (clip到范围内)

# 结构族权重（如果后续你还要混入别的结构，可扩展）
FAMILY_WEIGHTS = {
    "dbr_fp": 1.0,
}


# =========================
# 2) nk 加载（同 DBR：csv需包含 wl,n,k；wl 单位和 wavelengths_um 一致）
# =========================
def load_one_nk_csv(path: str, wavelengths_um: np.ndarray) -> np.ndarray:
    df = pd.read_csv(path).dropna()
    wl = df["wl"].values.astype(np.float64)
    n  = df["n"].values.astype(np.float64)
    k  = df["k"].values.astype(np.float64)

    k = np.clip(k, 0.0, None)

    n_interp = interp1d(wl, n, bounds_error=False, fill_value="extrapolate")
    k_interp = interp1d(wl, k, bounds_error=False, fill_value="extrapolate")

    n_i = n_interp(wavelengths_um)
    k_i = k_interp(wavelengths_um)

    # 防止插值负k
    k_i = np.clip(k_i, 0.0, None)

    # 关键：complex64 下给一个更稳的吸收底
    k_i[k_i < 1e-6] = 1e-6

    nk = (n_i + 1j * k_i).astype(np.complex64)
    return nk

def load_nk_torch(materials: List[str], wavelengths_um: np.ndarray) -> Dict[str, torch.Tensor]:
    nk: Dict[str, torch.Tensor] = {}

    # air
    nk_air = np.ones_like(wavelengths_um, dtype=np.complex64)
    nk["air"] = torch.from_numpy(nk_air).to(device=DEVICE, dtype=torch.complex64)

    for mat in materials:
        if mat == "air":
            continue

        path = os.path.join(NK_DIR, f"{mat}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing nk csv: {path}")

        nk_np = load_one_nk_csv(path, wavelengths_um)  # np.complex64
        nk[mat] = torch.from_numpy(nk_np).to(device=DEVICE, dtype=torch.complex64)

    return nk

def qw_thk_nm(nk_mat: torch.Tensor, wavelengths_um: np.ndarray, lambda0_um: float) -> int:
    """Quarter-wave thickness using real(n) at lambda0. For low-loss dielectrics this is fine."""
    idx0 = int(np.argmin(np.abs(wavelengths_um - lambda0_um)))
    n0 = float(np.real(nk_mat[idx0].detach().cpu().numpy()))
    t_nm = lambda0_um * 1000.0 / (4.0 * n0)
    return max(1, int(round(t_nm)))


# =========================
# 3) 结构生成：DBR + cavity + DBR
# =========================
def sample_family() -> str:
    return "dbr_fp"

def gen_dbr_fp(nk_dict_torch: Dict[str, torch.Tensor]) -> Tuple[List[str], List[int], dict]:
    H, L = random.choice(DBR_PAIRS)
    lambda0_um = float(random.choice(LAMBDA0_SET_UM))

    pairs_left = random.randint(PAIR_MIN, PAIR_MAX)
    if ALLOW_ASYM:
        delta = int(random.choice(ASYM_DELTA_SET))
        pairs_right = int(np.clip(pairs_left + delta, PAIR_MIN, PAIR_MAX))
    else:
        pairs_right = pairs_left

    cavity_mat = random.choice(CAVITY_MATS)
    mult = int(random.choice(CAVITY_MULT_SET))
    detune = float(random.choice(CAVITY_DETUNE_SET))

    # left DBR: (H L)^pairs_left
    mats: List[str] = []
    thks: List[int] = []
    for i in range(pairs_left * 2):
        m = H if (i % 2 == 0) else L
        mats.append(m)
        thks.append(qw_thk_nm(nk_dict_torch[m], WAVELENGTHS_UM, lambda0_um))

    # cavity: mult * QW * (1+detune)
    cav_qw = qw_thk_nm(nk_dict_torch[cavity_mat], WAVELENGTHS_UM, lambda0_um)
    cav_thk = int(round(mult * cav_qw * (1.0 + detune)))
    cav_thk = max(1, cav_thk)
    mats.append(cavity_mat)
    thks.append(cav_thk)

    # right DBR: mirror-symmetric tends to work well for high T
    # use (L H)^pairs_right so the interface around cavity is symmetric-ish
    for i in range(pairs_right * 2):
        m = L if (i % 2 == 0) else H
        mats.append(m)
        thks.append(qw_thk_nm(nk_dict_torch[m], WAVELENGTHS_UM, lambda0_um))

    meta = dict(
        family="dbr_fp",
        mirror_pair=f"{H}/{L}",
        lambda0_um=lambda0_um,
        pairs_left=int(pairs_left),
        pairs_right=int(pairs_right),
        cavity_mat=cavity_mat,
        cavity_mult=int(mult),
        cavity_detune=float(detune),
        num_layers=int(len(mats)),
        inc_medium=INC_MEDIUM,
        exit_medium=EXIT_MEDIUM,
    )
    return mats, thks, meta

def generate_one(nk_dict_torch):
    fam = sample_family()
    if fam == "dbr_fp":
        return gen_dbr_fp(nk_dict_torch)
    raise ValueError(f"Unknown family: {fam}")


# =========================
# 4) batch 打包 + tmm_fast
# =========================
def pack_batch_to_tmm_inputs(
    batch_mats: List[List[str]],
    batch_thks_nm: List[List[int]],
    nk_dict_torch: Dict[str, torch.Tensor],
    wl_m: torch.Tensor,
    inc_medium: str,
    exit_medium: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    tmm_fast.coh_tmm expects:
      n: (B, layers+2, W)
      d: (B, layers+2) with inf for semi-infinite media
    """
    B = len(batch_mats)
    W = wl_m.shape[0]
    Lmax = max(len(x) for x in batch_mats)

    d = torch.zeros((B, Lmax + 2), dtype=REAL_DTYPE, device=DEVICE)
    d[:, 0] = float("inf")
    d[:, -1] = float("inf")

    n = torch.empty((B, Lmax + 2, W), dtype=COMPLEX_DTYPE, device=DEVICE)

    n[:, 0, :] = nk_dict_torch[inc_medium]  # incident medium

    # exit medium may not exist => fallback to air
    if exit_medium in nk_dict_torch:
        n[:, -1, :] = nk_dict_torch[exit_medium]
    else:
        n[:, -1, :] = nk_dict_torch["air"]

    for bi in range(B):
        mats = batch_mats[bi]
        thks = batch_thks_nm[bi]
        L = len(mats)

        d[bi, 1:1 + L] = torch.tensor(thks, dtype=REAL_DTYPE, device=DEVICE) * 1e-9
        for li, m in enumerate(mats, start=1):
            n[bi, li, :] = nk_dict_torch[m]

        # padding：复制最后一层材料的nk（对 tmm_fast padding 稳定）
        if L < Lmax:
            n_pad = n[bi, L, :].clone()
            n[bi, 1 + L:1 + Lmax, :] = n_pad
    n = n.to(dtype=torch.complex64)
    d = d.to(dtype=torch.float32)
    return n, d

def calc_RT_fast_batch(
    batch_mats,
    batch_thks_nm,
    nk_dict_torch,
    wl_m,
    theta_rad,
    pol="s",
    inc_medium="air",
    exit_medium="Glass_Substrate",
):
    n, d = pack_batch_to_tmm_inputs(batch_mats, batch_thks_nm, nk_dict_torch, wl_m, inc_medium, exit_medium)

    print("[DBG] n dtype before tmm_fast:", n.dtype, "d dtype:", d.dtype,
        "theta dtype:", theta_rad.dtype, "wl dtype:", wl_m.dtype)
    print("[DBG] inc/exit n0:", n[0,0,0].item(), n[0,-1,0].item(), "theta:", theta_rad.item())

    out = tmm_fast.coh_tmm(pol, n, d, theta_rad, wl_m)
    R = out["R"]; T = out["T"]
    if R.ndim == 3:  # (B,1,W)
        R = R[:, 0, :]; T = T[:, 0, :]
    return (
        R.detach().cpu().float().numpy(),
        T.detach().cpu().float().numpy(),
    )


# =========================
# 5) 保存 split
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
# 6) 画图：叠加若干条曲线（T 与 R）
# =========================
def plot_overlay_curves(
    wavelengths_um: np.ndarray,
    R_list: List[np.ndarray],
    T_list: List[np.ndarray],
    save_dir: str,
    prefix: str = "overlay",
):
    os.makedirs(save_dir, exist_ok=True)

    # 透射叠图
    plt.figure()
    for T in T_list:
        plt.plot(wavelengths_um, T)
    plt.xlabel("Wavelength (um)")
    plt.ylabel("T")
    plt.title(f"Transmission overlay (N={len(T_list)})")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_T.png"), dpi=200)
    plt.close()

    # 反射叠图
    plt.figure()
    for R in R_list:
        plt.plot(wavelengths_um, R)
    plt.xlabel("Wavelength (um)")
    plt.ylabel("R")
    plt.title(f"Reflection overlay (N={len(R_list)})")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_R.png"), dpi=200)
    plt.close()

    print(f"[Plot] Saved overlay figures to: {save_dir}/{prefix}_T.png and {save_dir}/{prefix}_R.png")


# =========================
# 7) 主流程
# =========================
def main():
    print("DEVICE =", DEVICE, "| BATCH_SIZE =", BATCH_SIZE)

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_SEED)

    # 需要哪些材料
    needed = set(["air"])
    for H, L in DBR_PAIRS:
        needed.add(H); needed.add(L)
    for m in CAVITY_MATS:
        needed.add(m)

    # exit medium 如果存在就加载
    if EXIT_MEDIUM != "air":
        exit_csv = os.path.join(NK_DIR, f"{EXIT_MEDIUM}.csv")
        if os.path.exists(exit_csv):
            needed.add(EXIT_MEDIUM)
        else:
            print(f"[Warn] {EXIT_MEDIUM}.csv not found under {NK_DIR}. Fallback to air-air boundary.")

    missing = [m for m in sorted(needed) if (m != "air" and not os.path.exists(os.path.join(NK_DIR, f"{m}.csv")))]
    if missing:
        raise FileNotFoundError("Missing nk csv:\n  " + "\n  ".join(missing) + f"\nExpected under: {NK_DIR}")

    nk_dict_torch = load_nk_torch(sorted(list(needed)), WAVELENGTHS_UM)

    wl_m = torch.tensor(WAVELENGTHS_UM * 1e-6, dtype=torch.float32, device=DEVICE)
    #theta_rad = torch.tensor([0.0], dtype=torch.float32, device=DEVICE)
    theta_rad = torch.tensor([1e-3], dtype=torch.float32, device=DEVICE)  # ~0度但不在分支边界

    struct_list, spec_list, meta_list = [], [], []

    # 用于画图：保存一小撮样本的 R/T
    PLOT_N = 40  # 你可以改成 20/50/100
    plot_R, plot_T = [], []

    pbar = tqdm(total=NUM_SAMPLES, desc="Generating FP(DBR+cavity+DBR) dataset (tmm_fast batch)")
    while len(struct_list) < NUM_SAMPLES:
        cur = min(BATCH_SIZE, NUM_SAMPLES - len(struct_list))
        batch_mats, batch_thks, batch_meta = [], [], []

        for _ in range(cur):
            mats, thks, meta = generate_one(nk_dict_torch)
            batch_mats.append(mats)
            batch_thks.append(thks)
            batch_meta.append(meta)

        Rb, Tb = calc_RT_fast_batch(
            batch_mats, batch_thks, nk_dict_torch, wl_m, theta_rad,
            pol="s", inc_medium=INC_MEDIUM,
            exit_medium=(EXIT_MEDIUM if EXIT_MEDIUM in nk_dict_torch else "air")
        )

        for bi in range(cur):
            toks = [f"{m}_{int(t)}" for m, t in zip(batch_mats[bi], batch_thks[bi])]
            spec = np.concatenate([Rb[bi], Tb[bi]], axis=0).astype(np.float32).tolist()

            struct_list.append(toks)
            spec_list.append(spec)
            meta_list.append(batch_meta[bi])

            # 收集一些曲线用于叠图（从生成顺序里取前 PLOT_N 条即可）
            if len(plot_T) < PLOT_N:
                plot_R.append(Rb[bi].astype(np.float32))
                plot_T.append(Tb[bi].astype(np.float32))

        pbar.update(cur)

    pbar.close()
    save_split(struct_list, spec_list, meta_list, OUT_DIR)

    # quick stats
    lens = [len(x) for x in struct_list]
    print("\nstructure len: min/mean/max =", min(lens), sum(lens)/len(lens), max(lens))
    print("spec_dim =", len(spec_list[0]), "unique_spec_len =", len(set(len(x) for x in spec_list)))

    tok_cnt = Counter()
    for seq in struct_list:
        for t in seq:
            tok_cnt[t] += 1
    print("OBSERVED_UNIQUE_TOKENS_IN_DATA =", len(tok_cnt))
    print("Top tokens:", tok_cnt.most_common(10))

    # 画叠图
    plot_overlay_curves(WAVELENGTHS_UM, plot_R, plot_T, save_dir=OUT_DIR, prefix=f"overlay_{len(plot_T)}")

    # 再给一个很有用的快速指标：T_max 分布（粗看是否“峰顶能到接近1”）
    T_max = np.array([float(np.max(T)) for T in plot_T], dtype=np.float32)
    print(f"\n[QuickCheck] overlay samples T_max: min/mean/max = {T_max.min():.3f} / {T_max.mean():.3f} / {T_max.max():.3f}")
    print("Done.")


if __name__ == "__main__":
    main()
