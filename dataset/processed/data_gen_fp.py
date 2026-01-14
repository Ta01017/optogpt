#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_gen_fp.py

生成 Fabry-Perot (FP) 腔数据集（Structure/Spectrum），token = "Material_ThicknessNm"
- 使用 tmm_fast.coh_tmm (batch + wavelength vectorization)
- 结构来自业内常用 FP 组合：
  (A) Dielectric FP: DBR + cavity(defect) + DBR
  (B) MIM: Metal + cavity + Metal
  (C) Hybrid: DBR + cavity + Metal

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
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d

import torch
import tmm_fast


# =========================
# 0) 配置
# =========================
NUM_SAMPLES = 30000

# 波长网格
WAVELENGTHS_UM = np.linspace(0.9, 1.7, int(round((1.7 - 0.9) / 0.005)) + 1)
LAMBDA0_SET_UM = [0.95, 1.05, 1.20, 1.31, 1.55, 1.65]

NK_DIR = "./dataset/data1"
OUT_DIR = "./dataset/fp_cavity_smalltoken_gpu"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_RATIO = 0.8
SPLIT_SEED = 42
GLOBAL_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128 if torch.cuda.is_available() else 32

COMPLEX_DTYPE = torch.complex64
REAL_DTYPE = torch.float32


# =========================
# 1) 你的材料池（按你给的分类）
# =========================
SUBSTRATE = ["Glass_Substrate"]  # 作为基底：通常不当作 token 层（可选）

LOW_N  = ["MgF2", "SiO2", "ZnO"]
MID_N  = ["MgO", "Si3N4"]
HIGH_N = ["HfO2", "TiO2", "Ta2O5", "AlN", "Nb2O5", "ZnS", "ZnSe"]
UHIGH_N = ["Si", "a-Si"]
ABSORB  = ["ITO", "GaSb", "Ge"]
METAL   = ["Ag", "Au", "Al", "Al2O3", "TiN"]  # 注意：Al2O3严格说是介质，但你这里归到 metal，我照单全收


# =========================
# 2) 业内常见 DBR 镜材料对（你 DBR 那套直接复用）
# =========================
INDUSTRY_DBR_PAIRS: List[Tuple[str, str]] = [
    ("TiO2", "SiO2"),
    ("Ta2O5", "SiO2"),
    ("HfO2", "SiO2"),
    ("Nb2O5", "SiO2"),
    ("Si3N4", "SiO2"),
    ("AlN", "SiO2"),
]

# DBR 对数（每侧镜子的对数）
DBR_PAIR_MIN = 4
DBR_PAIR_MAX = 8

# FP 腔阶次（半波条件 t = m * λ0/(2n)）
CAVITY_ORDER_SET = [1, 2, 3]     # 可加到 [1..5]，token 会更多

# MIM 金属层厚度离散集合（nm）——业内常用：10~60nm 量级
METAL_THK_SET_NM = [10, 15, 20, 25, 30, 40, 50, 60]

# 选择哪些材料当腔层（常见：SiO2/MgF2/Si3N4/MgO/ITO 等）
CAVITY_MATS = ["SiO2", "MgF2", "Si3N4", "MgO", "ITO"]


# =========================
# 3) 三类 FP 组合的采样权重（可调）
# =========================
FAMILY_WEIGHTS = {
    "dielectric_dbr": 0.55,  # DBR + cavity + DBR
    "mim_metal":      0.30,  # Metal + cavity + Metal
    "hybrid_dbr_m":   0.15,  # DBR + cavity + Metal
}


# =========================
# 4) 加载 nk（wl,n,k csv -> 插值到 WAVELENGTHS_UM），转 torch 放 DEVICE
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
# 5) 离散厚度表：quarter-wave / half-wave(m阶)
# =========================
def precompute_qw_table_nm(nk_dict_torch: Dict[str, torch.Tensor],
                           wavelengths_um: np.ndarray,
                           lambda0_set_um: List[float]) -> Dict[str, Dict[int, int]]:
    wl = wavelengths_um
    thk_table: Dict[str, Dict[int, int]] = {}
    for mat, nk_arr_t in nk_dict_torch.items():
        thk_table[mat] = {}
        nk_arr = nk_arr_t.detach().cpu().numpy()
        for lam0 in lambda0_set_um:
            idx0 = int(np.argmin(np.abs(wl - lam0)))
            n0 = float(np.real(nk_arr[idx0]))
            t_nm = lam0 * 1000.0 / (4.0 * n0)          # quarter-wave
            thk_table[mat][int(round(lam0 * 1000))] = int(round(t_nm))
    return thk_table


def precompute_hw_table_nm(nk_dict_torch: Dict[str, torch.Tensor],
                           wavelengths_um: np.ndarray,
                           lambda0_set_um: List[float],
                           order_set: List[int]) -> Dict[str, Dict[Tuple[int, int], int]]:
    wl = wavelengths_um
    thk_table: Dict[str, Dict[Tuple[int, int], int]] = {}
    for mat, nk_arr_t in nk_dict_torch.items():
        thk_table[mat] = {}
        nk_arr = nk_arr_t.detach().cpu().numpy()
        for lam0 in lambda0_set_um:
            idx0 = int(np.argmin(np.abs(wl - lam0)))
            n0 = float(np.real(nk_arr[idx0]))
            for m in order_set:
                t_nm = (m * lam0 * 1000.0) / (2.0 * n0)   # m * half-wave
                thk_table[mat][(int(round(lam0 * 1000)), int(m))] = int(round(t_nm))
    return thk_table


# =========================
# 6) 三类 FP 结构生成
# =========================
def sample_family() -> str:
    keys = list(FAMILY_WEIGHTS.keys())
    probs = np.array([FAMILY_WEIGHTS[k] for k in keys], dtype=np.float64)
    probs = probs / probs.sum()
    return np.random.choice(keys, p=probs).item()


def gen_dielectric_dbr_fp(qw_table: Dict[str, Dict[int, int]],
                          hw_table: Dict[str, Dict[Tuple[int, int], int]]) -> Tuple[List[str], List[int], dict]:
    H, L = random.choice(INDUSTRY_DBR_PAIRS)
    lambda0_um = random.choice(LAMBDA0_SET_UM)
    lambda0_nm = int(round(lambda0_um * 1000))
    pairs = random.randint(DBR_PAIR_MIN, DBR_PAIR_MAX)

    cav_mat = random.choice(CAVITY_MATS)
    cav_order = random.choice(CAVITY_ORDER_SET)
    cav_thk = hw_table[cav_mat][(lambda0_nm, cav_order)]

    # 左镜：(H L)^pairs
    mats, thks = [], []
    for i in range(pairs * 2):
        mat = H if (i % 2 == 0) else L
        mats.append(mat)
        thks.append(qw_table[mat][lambda0_nm])

    # cavity
    mats.append(cav_mat)
    thks.append(cav_thk)

    # 右镜：镜像 (L H)^pairs（常见对称结构）
    for i in range(pairs * 2):
        mat = L if (i % 2 == 0) else H
        mats.append(mat)
        thks.append(qw_table[mat][lambda0_nm])

    meta = dict(
        family="dielectric_dbr",
        mirror_pair=f"{H}/{L}",
        lambda0_um=float(lambda0_um),
        lambda0_nm=int(lambda0_nm),
        dbr_pairs_each_side=int(pairs),
        cavity_mat=cav_mat,
        cavity_order=int(cav_order),
        num_layers=int(len(mats)),
    )
    return mats, thks, meta


def gen_mim_fp(hw_table: Dict[str, Dict[Tuple[int, int], int]]) -> Tuple[List[str], List[int], dict]:
    # 常见金属镜：Ag/Au/Al/TiN（可按你的 nk 覆盖情况删）
    metal = random.choice(["Ag", "Au", "Al", "TiN"])
    t_metal_1 = random.choice(METAL_THK_SET_NM)
    t_metal_2 = random.choice(METAL_THK_SET_NM)

    lambda0_um = random.choice(LAMBDA0_SET_UM)
    lambda0_nm = int(round(lambda0_um * 1000))
    cav_mat = random.choice(CAVITY_MATS)
    cav_order = random.choice(CAVITY_ORDER_SET)
    cav_thk = hw_table[cav_mat][(lambda0_nm, cav_order)]

    mats = [metal, cav_mat, metal]
    thks = [t_metal_1, cav_thk, t_metal_2]

    meta = dict(
        family="mim_metal",
        metal=metal,
        lambda0_um=float(lambda0_um),
        lambda0_nm=int(lambda0_nm),
        cavity_mat=cav_mat,
        cavity_order=int(cav_order),
        metal_thk_nm=[int(t_metal_1), int(t_metal_2)],
        num_layers=int(len(mats)),
    )
    return mats, thks, meta


def gen_hybrid_fp(qw_table: Dict[str, Dict[int, int]],
                  hw_table: Dict[str, Dict[Tuple[int, int], int]]) -> Tuple[List[str], List[int], dict]:
    H, L = random.choice(INDUSTRY_DBR_PAIRS)
    metal = random.choice(["Ag", "Au", "Al", "TiN"])

    lambda0_um = random.choice(LAMBDA0_SET_UM)
    lambda0_nm = int(round(lambda0_um * 1000))
    pairs = random.randint(DBR_PAIR_MIN, DBR_PAIR_MAX)

    cav_mat = random.choice(CAVITY_MATS)
    cav_order = random.choice(CAVITY_ORDER_SET)
    cav_thk = hw_table[cav_mat][(lambda0_nm, cav_order)]

    t_metal = random.choice(METAL_THK_SET_NM)

    mats, thks = [], []
    # DBR 一侧：(H L)^pairs
    for i in range(pairs * 2):
        mat = H if (i % 2 == 0) else L
        mats.append(mat)
        thks.append(qw_table[mat][lambda0_nm])

    # cavity
    mats.append(cav_mat)
    thks.append(cav_thk)

    # metal mirror
    mats.append(metal)
    thks.append(t_metal)

    meta = dict(
        family="hybrid_dbr_m",
        mirror_pair=f"{H}/{L}",
        metal=metal,
        lambda0_um=float(lambda0_um),
        lambda0_nm=int(lambda0_nm),
        dbr_pairs=int(pairs),
        cavity_mat=cav_mat,
        cavity_order=int(cav_order),
        metal_thk_nm=int(t_metal),
        num_layers=int(len(mats)),
    )
    return mats, thks, meta


def generate_fp_structure(qw_table, hw_table):
    fam = sample_family()
    if fam == "dielectric_dbr":
        return gen_dielectric_dbr_fp(qw_table, hw_table)
    if fam == "mim_metal":
        return gen_mim_fp(hw_table)
    if fam == "hybrid_dbr_m":
        return gen_hybrid_fp(qw_table, hw_table)
    raise RuntimeError("Unknown family: " + str(fam))


# =========================
# 7) batch 打包：pad 到 batch 内最大层数
# =========================
def pack_batch_to_tmm_inputs(batch_mats: List[List[str]],
                             batch_thks_nm: List[List[int]],
                             nk_dict_torch: Dict[str, torch.Tensor],
                             wl_m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回:
      n: [B, Lmax+2, num_wl] complex
      d: [B, Lmax+2] real (meters, with inf at ends)
    约定：入射介质=空气(1.0)，出射介质=空气(1.0)
    若你希望出射为 BK5 基底：把 n[:, -1, :] 改成 nk_dict_torch['Glass_Substrate']
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

        d[bi, 1:1 + L] = torch.tensor(thks, dtype=REAL_DTYPE, device=DEVICE) * 1e-9
        for li, m in enumerate(mats, start=1):
            n[bi, li, :] = nk_dict_torch[m]

        if L < Lmax:
            # pad：厚度=0；n 复制最后一层（避免奇怪的数值）
            n_pad_val = n[bi, L, :].clone()
            n[bi, 1 + L:1 + Lmax, :] = n_pad_val

    return n, d


# =========================
# 8) tmm_fast batch 计算
# =========================
def calc_RT_fast_batch(batch_mats, batch_thks_nm, nk_dict_torch, wl_m, theta_rad, pol="s"):
    n, d = pack_batch_to_tmm_inputs(batch_mats, batch_thks_nm, nk_dict_torch, wl_m)
    out = tmm_fast.coh_tmm(pol, n, d, theta_rad, wl_m)
    R = out["R"]
    T = out["T"]
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
# 9) 保存 train/dev
# =========================
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
# 10) 主流程
# =========================
def main():
    print("DEVICE =", DEVICE, "| BATCH_SIZE =", BATCH_SIZE)

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_SEED)

    # 统计本脚本可能用到的材料，检查 nk 是否齐全
    needed = set()
    for H, L in INDUSTRY_DBR_PAIRS:
        needed.add(H); needed.add(L)
    for m in CAVITY_MATS:
        needed.add(m)
    for m in ["Ag", "Au", "Al", "TiN"]:
        needed.add(m)
    # 如果你想让出射介质为 BK5：需要加 Glass_Substrate
    # needed.add("Glass_Substrate")

    missing = [m for m in sorted(needed) if not os.path.exists(os.path.join(NK_DIR, f"{m}.csv"))]
    if missing:
        raise FileNotFoundError("Missing nk csv:\n  " + "\n  ".join(missing) + f"\nExpected under: {NK_DIR}")

    wl_m = torch.tensor(WAVELENGTHS_UM * 1e-6, dtype=REAL_DTYPE, device=DEVICE)
    theta_rad = torch.tensor([0.0], dtype=REAL_DTYPE, device=DEVICE)

    nk_dict_torch = load_nk_torch(sorted(list(needed)), WAVELENGTHS_UM)

    qw_table = precompute_qw_table_nm(nk_dict_torch, WAVELENGTHS_UM, LAMBDA0_SET_UM)
    hw_table = precompute_hw_table_nm(nk_dict_torch, WAVELENGTHS_UM, LAMBDA0_SET_UM, CAVITY_ORDER_SET)

    # ========== 生成 ==========
    struct_list, spec_list, meta_list = [], [], []
    pbar = tqdm(total=NUM_SAMPLES, desc="Generating FP cavity dataset (tmm_fast batch)")

    while len(struct_list) < NUM_SAMPLES:
        cur = min(BATCH_SIZE, NUM_SAMPLES - len(struct_list))
        batch_mats, batch_thks, batch_meta = [], [], []

        for _ in range(cur):
            mats, thks, meta = generate_fp_structure(qw_table, hw_table)
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

    # ========== 统计：token 分布 + “token 类别总数” ==========
    tok_cnt = Counter()
    fam_cnt = Counter()
    for seq, meta in zip(struct_list, meta_list):
        fam_cnt[meta["family"]] += 1
        for tok in seq:
            tok_cnt[tok] += 1

    print("\n== Family distribution ==")
    total = sum(fam_cnt.values())
    for k, v in fam_cnt.most_common():
        print(f"  {k:18s} {v:6d} ({v/total:.3%})")

    print("\n== Top tokens (raw occurrences) ==")
    tot_tok = sum(tok_cnt.values())
    for t, c in tok_cnt.most_common(20):
        print(f"  {t:18s} {c:7d} ({c/tot_tok:.3%})")

    # ---- 精确计算“理论 token 空间”（由离散厚度表定义）----
    # DBR：用 quarter-wave 表（只对可能出现在 DBR 的材料统计）
    possible_thk = defaultdict(set)

    # DBR 对材料（qw）
    dbr_mats = set()
    for H, L in INDUSTRY_DBR_PAIRS:
        dbr_mats.add(H); dbr_mats.add(L)
    for mat in dbr_mats:
        for lam0_nm, t in qw_table[mat].items():
            possible_thk[mat].add(int(t))

    # cavity 材料（hw）
    for mat in CAVITY_MATS:
        for (lam0_nm, order), t in hw_table[mat].items():
            possible_thk[mat].add(int(t))

    # 金属厚度（固定离散集合）
    for mat in ["Ag", "Au", "Al", "TiN"]:
        for t in METAL_THK_SET_NM:
            possible_thk[mat].add(int(t))

    # 计算总 token 类别数
    vocab_tokens = []
    for mat, thks in possible_thk.items():
        for t in sorted(thks):
            vocab_tokens.append(f"{mat}_{t}")

    print("\n== Token space summary (theoretical by design) ==")
    mats_sorted = sorted(possible_thk.keys())
    total_vocab = 0
    for mat in mats_sorted:
        k = len(possible_thk[mat])
        total_vocab += k
        print(f"  {mat:10s} unique_thk={k:3d}  thk_nm={sorted(possible_thk[mat])[:10]}{'...' if k>10 else ''}")

    print("\n== TOTAL token categories (unique Material_ThicknessNm) ==")
    print("TOTAL_VOCAB =", total_vocab)

    # 也给你“数据里实际出现过的 unique token 数量”
    print("OBSERVED_UNIQUE_TOKENS_IN_DATA =", len(tok_cnt))

    print("\nDone. OUT_DIR =", OUT_DIR)


if __name__ == "__main__":
    main()
