#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_dbr_small_token_30k_tmm_fast_batch_with_validation.py

- 用你安装的 tmm_fast（导出 coh_tmm / inc_tmm）替代 tmm
- 向量化：
  * 波长维度：一次性算完整 wavelength 网格
  * 样本维度：batch 计算（一次算 B 个结构）
- 生成 DBR 数据集（Structure/Spectrum），结构 token 数量很小：
  * 不使用随机厚度扰动
  * 厚度只来源于少量中心波长 lambda0 的 quarter-wave 厚度
  * token 仍然为 "Material_ThicknessNm"，但每种材料只有 ~len(LAMBDA0_SET) 种厚度

- 内置验证（强烈建议先开验证再跑满 30k）：
  1) tmm_fast vs tmm（参考逐波长 coh_tmm）数值一致性：max|dR|/RMSE 等
  2) DBR 行为 sanity：R(lambda0)、stopband 宽度估计
  3) 能量守恒 sanity：R+T <= 1 (+eps)

输出：
OUT_DIR/
  Structure_train.pkl
  Spectrum_train.pkl
  Structure_dev.pkl
  Spectrum_dev.pkl
  meta_train.pkl
  meta_dev.pkl

Spectrum 每条为 [R..., T...]，波长网格：0.9~1.7 um, step=0.005
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
import tmm_fast  # 你这个版本导出：coh_tmm / inc_tmm / plot_stacks
from tmm import coh_tmm as coh_tmm_ref  # 用于验证（逐波长参考实现）


# =========================
# 0) 配置
# =========================
NUM_SAMPLES = 30000

PAIR_MIN = 6
PAIR_MAX = 10

# 波长网格（与你原代码一致）
WAVELENGTHS_UM = np.linspace(0.9, 1.7, int(round((1.7 - 0.9) / 0.005)) + 1)

# 少量中心波长 lambda0（决定厚度 token 数量）
LAMBDA0_SET_UM = [0.95, 1.05, 1.20, 1.31, 1.55, 1.65]

NK_DIR = "./dataset/data"
OUT_DIR = "./dataset/dbr_smalltoken_gpu"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_RATIO = 0.8
SPLIT_SEED = 42
GLOBAL_SEED = 42

# tmm_fast 基于 torch，支持 GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# batch 大小（一次算多少个结构）
BATCH_SIZE = 128 if torch.cuda.is_available() else 32

# dtype
COMPLEX_DTYPE = torch.complex64
REAL_DTYPE = torch.float32


# =========================
# 1) 业内常用材料对
# =========================
INDUSTRY_CORE: List[Tuple[str, str]] = [
    ("TiO2", "SiO2"),
    ("Ta2O5", "SiO2"),
    ("HfO2", "SiO2"),
    ("Nb2O5", "SiO2"),
    ("Si3N4", "SiO2"),
    ("AlN", "SiO2"),
]


# =========================
# 2) 加载 nk（wl,n,k csv -> 插值到 WAVELENGTHS_UM）
#    转 torch，放 DEVICE
# =========================
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
        nk_t = torch.tensor(nk_np, dtype=COMPLEX_DTYPE, device=DEVICE)
        nk[mat] = nk_t
    return nk


# =========================
# 3) 预计算离散厚度表（quarter-wave）
# =========================
def precompute_qw_thickness_table(
    nk_dict_torch: Dict[str, torch.Tensor],
    wavelengths_um: np.ndarray,
    lambda0_set_um: List[float],
) -> Dict[str, Dict[int, int]]:
    wl = wavelengths_um
    thk_table: Dict[str, Dict[int, int]] = {}
    for mat, nk_arr_t in nk_dict_torch.items():
        thk_table[mat] = {}
        nk_arr = nk_arr_t.detach().cpu().numpy()
        for lam0 in lambda0_set_um:
            idx0 = int(np.argmin(np.abs(wl - lam0)))
            n0 = float(np.real(nk_arr[idx0]))
            t_nm = lam0 * 1000.0 / (4.0 * n0)
            t_nm_int = int(round(t_nm))
            thk_table[mat][int(round(lam0 * 1000))] = t_nm_int
    return thk_table


# =========================
# 4) 生成一个 DBR（离散厚度）
# =========================
def generate_dbr_discrete(
    industry_pairs: List[Tuple[str, str]],
    thk_table: Dict[str, Dict[int, int]],
) -> Tuple[List[str], List[int], dict]:
    H, L = random.choice(industry_pairs)
    lambda0_um = random.choice(LAMBDA0_SET_UM)
    lambda0_nm = int(round(lambda0_um * 1000))
    pairs = random.randint(PAIR_MIN, PAIR_MAX)

    materials, thicknesses = [], []
    for i in range(pairs * 2):
        mat = H if (i % 2 == 0) else L
        t_nm = thk_table[mat][lambda0_nm]
        materials.append(mat)
        thicknesses.append(t_nm)

    meta = dict(
        pair_H=H,
        pair_L=L,
        pair_name=f"{H}/{L}",
        lambda0_um=float(lambda0_um),
        lambda0_nm=int(lambda0_nm),
        pairs=int(pairs),
        num_layers=int(pairs * 2),
        mode="discrete_qw",
    )
    return materials, thicknesses, meta


# =========================
# 5) batch 打包：pad 到 batch 内最大层数
# =========================
def pack_batch_to_tmm_inputs(
    batch_mats: List[List[str]],
    batch_thks_nm: List[List[int]],
    nk_dict_torch: Dict[str, torch.Tensor],
    wl_m: torch.Tensor,  # [num_wl]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回:
      n: [B, Lmax+2, num_wl] complex
      d: [B, Lmax+2] real (meters, with inf at ends)
    """
    B = len(batch_mats)
    num_wl = wl_m.shape[0]
    Lmax = max(len(x) for x in batch_mats)

    d = torch.zeros((B, Lmax + 2), dtype=REAL_DTYPE, device=DEVICE)
    d[:, 0] = float("inf")
    d[:, -1] = float("inf")

    n = torch.empty((B, Lmax + 2, num_wl), dtype=COMPLEX_DTYPE, device=DEVICE)
    n[:, 0, :] = (1.0 + 0.0j)
    n[:, -1, :] = (1.0 + 0.0j)

    for bi in range(B):
        mats = batch_mats[bi]
        thks = batch_thks_nm[bi]
        L = len(mats)

        d[bi, 1:1 + L] = torch.tensor(thks, dtype=REAL_DTYPE, device=DEVICE) * 1e-9  # nm -> m
        for li, m in enumerate(mats, start=1):
            n[bi, li, :] = nk_dict_torch[m]

        if L < Lmax:
            # pad 层：厚度=0；n 复制最后一层（也可以设 1.0）
            n_pad_val = n[bi, L, :].clone()  # 注意：L 层对应索引 L（因为前面有入射介质）
            n[bi, 1 + L:1 + Lmax, :] = n_pad_val

    return n, d


# =========================
# 6) tmm_fast batch 计算（波长 + 样本向量化）
# =========================
def calc_RT_fast_batch(
    batch_mats: List[List[str]],
    batch_thks_nm: List[List[int]],
    nk_dict_torch: Dict[str, torch.Tensor],
    wl_m: torch.Tensor,       # [num_wl], meters
    theta_rad: torch.Tensor,  # [A]
    pol: str = "s",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回:
      R: [B, num_wl] float32 numpy
      T: [B, num_wl] float32 numpy
    """
    n, d = pack_batch_to_tmm_inputs(batch_mats, batch_thks_nm, nk_dict_torch, wl_m)

    # 你的 tmm_fast 版本导出 coh_tmm（色散、多 stack、向量化）
    out = tmm_fast.coh_tmm(pol, n, d, theta_rad, wl_m)

    R = out["R"]
    T = out["T"]

    # 统一到 [B, num_wl]
    if R.ndim == 3:
        R = R[:, 0, :]
        T = T[:, 0, :]
    elif R.ndim != 2:
        raise RuntimeError(f"Unexpected R shape: {tuple(R.shape)}")

    return (
        R.detach().to("cpu").float().numpy(),
        T.detach().to("cpu").float().numpy(),
    )


# =========================
# 7) 保存 train/dev
# =========================
def save_split(struct_list, spec_list, meta_list, out_dir):
    N = len(struct_list)
    rng = np.random.default_rng(SPLIT_SEED)
    idx = rng.permutation(N)
    split = int(N * TRAIN_RATIO)
    idx_train = idx[:split]
    idx_dev = idx[split:]

    train_struct = [struct_list[i] for i in idx_train]
    train_spec = [spec_list[i] for i in idx_train]
    train_meta = [meta_list[i] for i in idx_train]

    dev_struct = [struct_list[i] for i in idx_dev]
    dev_spec = [spec_list[i] for i in idx_dev]
    dev_meta = [meta_list[i] for i in idx_dev]

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
    print(" -", os.path.join(out_dir, "Structure_train.pkl"))
    print(" -", os.path.join(out_dir, "Spectrum_train.pkl"))
    print(" -", os.path.join(out_dir, "Structure_dev.pkl"))
    print(" -", os.path.join(out_dir, "Spectrum_dev.pkl"))
    print(" -", os.path.join(out_dir, "meta_train.pkl"))
    print(" -", os.path.join(out_dir, "meta_dev.pkl"))


# =========================
# 8) 验证：参考 tmm（逐波长） vs tmm_fast（向量化）
# =========================
def calc_RT_tmm_reference(materials, thicknesses_nm, nk_dict_np, wavelengths_um, pol="s", theta_deg=0.0):
    """
    参考实现：逐波长循环 + tmm.coh_tmm
    注意：
      - tmm 的 lam_vac 可用 nm 或任意一致单位，只要 n_list 是对应波长处折射率
      - 这里 thickness 用 nm（与你结构一致），lam_vac 用 nm
    """
    th0 = np.deg2rad(theta_deg)
    d_list = [np.inf] + list(map(float, thicknesses_nm)) + [np.inf]

    R = np.zeros_like(wavelengths_um, dtype=np.float64)
    T = np.zeros_like(wavelengths_um, dtype=np.float64)

    wl_nm_list = (wavelengths_um * 1000.0).astype(np.float64)
    for i, lam_nm in enumerate(wl_nm_list):
        n_list = [1.0] + [nk_dict_np[m][i] for m in materials] + [1.0]
        res = coh_tmm_ref(pol=pol, n_list=n_list, d_list=d_list, th_0=th0, lam_vac=lam_nm)
        R[i] = float(res["R"])
        T[i] = float(res["T"])
    return R.astype(np.float32), T.astype(np.float32)


def stopband_metrics(R, wavelengths_um, lambda0_um):
    w = wavelengths_um
    i0 = int(np.argmin(np.abs(w - lambda0_um)))

    thr = 0.90
    mask = (R >= thr)
    if mask.sum() == 0:
        thr = 0.80
        mask = (R >= thr)

    band_width = 0.0
    band_mean_R = float(np.mean(R[mask])) if mask.sum() > 0 else float(np.mean(R))

    if mask[i0]:
        l = i0
        while l - 1 >= 0 and mask[l - 1]:
            l -= 1
        r = i0
        while r + 1 < len(mask) and mask[r + 1]:
            r += 1
        band_width = float(w[r] - w[l])
        band_mean_R = float(np.mean(R[l:r + 1]))
    else:
        win = 5
        l = max(0, i0 - win)
        r = min(len(R) - 1, i0 + win)
        band_mean_R = float(np.mean(R[l:r + 1]))
        band_width = 0.0

    R0 = float(R[i0])
    return {
        "R_at_lambda0": R0,
        "stopband_width_um_est": band_width,
        "stopband_mean_R_est": band_mean_R,
        "threshold_used": thr,
    }


def validate_dbr(
    nk_dict_torch: Dict[str, torch.Tensor],
    wavelengths_um: np.ndarray,
    thk_table: Dict[str, Dict[int, int]],
    industry_pairs: List[Tuple[str, str]],
    num_checks: int = 10,
    seed: int = 123,
    plot: bool = False,
    eps_energy: float = 2e-4,
):
    """
    1) tmm_fast vs tmm（参考逐波长 coh_tmm）数值一致性
    2) DBR 行为 sanity：R(lambda0)、stopband 宽度估计
    3) 能量守恒 sanity：R+T <= 1 + eps
    """
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)

    nk_dict_np = {k: v.detach().cpu().numpy() for k, v in nk_dict_torch.items()}

    wl_m = torch.tensor(wavelengths_um * 1e-6, dtype=REAL_DTYPE, device=DEVICE)
    theta_rad = torch.tensor([0.0], dtype=REAL_DTYPE, device=DEVICE)

    print("\n================ VALIDATION ================")
    print(f"num_checks={num_checks} | plot={plot} | energy_eps={eps_energy:g}")

    err_R = []
    err_T = []
    energy_viol = []
    R0s = []
    bws = []

    for ci in range(num_checks):
        mats, thks, meta = generate_dbr_discrete(industry_pairs, thk_table)
        lambda0_um = meta["lambda0_um"]

        # tmm_fast（batch=1）
        R_fast, T_fast = calc_RT_fast_batch(
            batch_mats=[mats],
            batch_thks_nm=[thks],
            nk_dict_torch=nk_dict_torch,
            wl_m=wl_m,
            theta_rad=theta_rad,
            pol="s",
        )
        R_fast = R_fast[0]
        T_fast = T_fast[0]

        # reference（逐点 tmm）
        R_ref, T_ref = calc_RT_tmm_reference(
            mats, thks, nk_dict_np, wavelengths_um, pol="s", theta_deg=0.0
        )

        dR = R_fast - R_ref
        dT = T_fast - T_ref
        max_abs_R = float(np.max(np.abs(dR)))
        rmse_R = float(np.sqrt(np.mean(dR ** 2)))
        max_abs_T = float(np.max(np.abs(dT)))
        rmse_T = float(np.sqrt(np.mean(dT ** 2)))

        err_R.append((max_abs_R, rmse_R))
        err_T.append((max_abs_T, rmse_T))

        # 能量守恒 sanity
        rt_sum = R_fast + T_fast
        viol = float(np.max(rt_sum - 1.0))
        energy_viol.append(viol)

        # DBR 指标
        m = stopband_metrics(R_fast, wavelengths_um, lambda0_um)
        R0s.append(m["R_at_lambda0"])
        bws.append(m["stopband_width_um_est"])

        print(
            f"[{ci:02d}] {meta['pair_name']} pairs={meta['pairs']} lambda0={lambda0_um:.3f}um | "
            f"max|dR|={max_abs_R:.3e} rmseR={rmse_R:.3e} | max|dT|={max_abs_T:.3e} rmseT={rmse_T:.3e} | "
            f"max(R+T-1)={viol:.3e} | R0={m['R_at_lambda0']:.3f} bandW~{m['stopband_width_um_est']:.3f}"
        )

        if plot:
            plt.figure()
            plt.plot(wavelengths_um, R_ref, label="R_ref(tmm)")
            plt.plot(wavelengths_um, R_fast, "--", label="R_fast(tmm_fast)")
            plt.axvline(lambda0_um, linestyle=":", label="lambda0")
            plt.ylim(-0.05, 1.05)
            plt.xlabel("Wavelength (um)")
            plt.ylabel("Reflectance R")
            plt.title(f"{meta['pair_name']} pairs={meta['pairs']} lambda0={lambda0_um:.3f}um")
            plt.legend()
            plt.show()

    max_abs_R_all = max(x[0] for x in err_R)
    rmse_R_mean = float(np.mean([x[1] for x in err_R]))
    max_abs_T_all = max(x[0] for x in err_T)
    rmse_T_mean = float(np.mean([x[1] for x in err_T]))

    worst_energy = float(np.max(energy_viol))

    print("\n---- Summary (tmm_fast vs tmm) ----")
    print(f"R: worst max|dR|={max_abs_R_all:.3e}, mean rmseR={rmse_R_mean:.3e}")
    print(f"T: worst max|dT|={max_abs_T_all:.3e}, mean rmseT={rmse_T_mean:.3e}")

    print("\n---- Energy sanity ----")
    print(f"worst max(R+T-1) = {worst_energy:.3e}")
    if worst_energy > eps_energy:
        print("WARNING: energy violation seems large. Check units/shapes/dtypes.")
    else:
        print("OK: energy violation within tolerance (absorption can make R+T<1, but should not exceed 1 much).")

    print("\n---- DBR sanity (using tmm_fast R) ----")
    print(f"R(lambda0): mean={float(np.mean(R0s)):.3f}, min={float(np.min(R0s)):.3f}, max={float(np.max(R0s)):.3f}")
    print(f"stopband_width_est(um): mean={float(np.mean(bws)):.3f}, min={float(np.min(bws)):.3f}, max={float(np.max(bws)):.3f}")
    print("==========================================\n")


# =========================
# 9) 主流程
# =========================
def main():
    print("DEVICE =", DEVICE, "| BATCH_SIZE =", BATCH_SIZE)

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_SEED)

    # 检查 nk 文件
    needed = set()
    for H, L in INDUSTRY_CORE:
        needed.add(H)
        needed.add(L)

    missing = [m for m in sorted(needed) if not os.path.exists(os.path.join(NK_DIR, f"{m}.csv"))]
    if missing:
        raise FileNotFoundError("Missing nk csv:\n  " + "\n  ".join(missing) + f"\nExpected under: {NK_DIR}")

    # wl/theta torch（SI）
    wl_m = torch.tensor(WAVELENGTHS_UM * 1e-6, dtype=REAL_DTYPE, device=DEVICE)  # [num_wl]
    theta_rad = torch.tensor([0.0], dtype=REAL_DTYPE, device=DEVICE)             # [1]

    # nk -> torch on DEVICE
    nk_dict_torch = load_nk_torch(sorted(list(needed)), WAVELENGTHS_UM)

    # 厚度表
    thk_table = precompute_qw_thickness_table(nk_dict_torch, WAVELENGTHS_UM, LAMBDA0_SET_UM)

    print("\n== Discrete thickness table (nm) ==")
    for m in sorted(thk_table.keys()):
        vals = sorted(set(thk_table[m].values()))
        print(f"{m:8s} unique_thk={len(vals):2d}  values={vals}")

    # ==========（可选但强烈建议）先做验证 ==========
    # 你可以先 num_checks=5，确认误差很小，再跑满 30k
    validate_dbr(
        nk_dict_torch=nk_dict_torch,
        wavelengths_um=WAVELENGTHS_UM,
        thk_table=thk_table,
        industry_pairs=INDUSTRY_CORE,
        num_checks=10,
        seed=123,
        plot=False,
        eps_energy=2e-4,
    )

    # ========== 生成数据 ==========
    struct_list, spec_list, meta_list = [], [], []

    pbar = tqdm(total=NUM_SAMPLES, desc="Generating DBR small-token 30k (tmm_fast coh_tmm batch)")
    while len(struct_list) < NUM_SAMPLES:
        cur = min(BATCH_SIZE, NUM_SAMPLES - len(struct_list))
        batch_mats, batch_thks, batch_meta = [], [], []

        for _ in range(cur):
            mats, thks, meta = generate_dbr_discrete(INDUSTRY_CORE, thk_table)
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
        )  # [cur, num_wl]

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

    # 保存 train/dev
    save_split(struct_list, spec_list, meta_list, OUT_DIR)

    # ========== 统计 ==========
    pair_cnt = Counter()
    cover_cnt = Counter()
    tok_cnt = Counter()

    for seq, meta in zip(struct_list, meta_list):
        pair_cnt[meta["pair_name"]] += 1
        cover_cnt[(meta["pair_name"], meta["lambda0_nm"], meta["pairs"])] += 1
        for tok in seq:
            tok_cnt[tok] += 1

    print("\n== Pair distribution ==")
    total = sum(pair_cnt.values())
    for k, v in pair_cnt.most_common():
        print(f"  {k:15s} {v:6d} ({v/total:.3%})")

    print("\n== Coverage (pair_name, lambda0_nm, pairs) top 30 ==")
    for k, v in cover_cnt.most_common(30):
        print(" ", k, v)
    print("unique (pair,lambda0,pairs) combos:", len(cover_cnt))

    print("\n== Top tokens (raw) ==")
    tot_tok = sum(tok_cnt.values())
    for t, c in tok_cnt.most_common(20):
        print(f"  {t:15s} {c:6d} ({c/tot_tok:.3%})")

    mats = sorted(list(needed))
    approx_unique_thk = {m: len(set(thk_table[m].values())) for m in mats}
    approx_vocab = sum(approx_unique_thk.values())
    print("\n== Approx token space ==")
    print("materials:", mats)
    print("unique_thk per material:", approx_unique_thk)
    print("approx total structure tokens (material_thk):", approx_vocab)

    print("\nDone. OUT_DIR =", OUT_DIR)


if __name__ == "__main__":
    main()
