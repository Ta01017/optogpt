#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_multicavity_smalltoken_tmm_fast_batch.py

多腔滤波器（Multi-Cavity, 2C/3C）数据生成：
- token = "Material_ThicknessNm"
- 结构（业内常见）： EndDBR + cavity + (MidDBR + cavity)* + EndDBR
- EndDBR pairs: 较大（高反端镜）
- MidDBR pairs: 较小（耦合镜）
- cavity: half-wave(m阶) 离散厚度
- DBR: quarter-wave 离散厚度
- 使用 tmm_fast.coh_tmm：batch + wavelength vectorization

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

# 64
# =========================
# 0) 配置
# =========================
NUM_SAMPLES = 100000

WAVELENGTHS_UM = np.linspace(0.9, 1.7, int(round((1.7 - 0.9) / 0.005)) + 1)
LAMBDA0_SET_UM = [0.95, 1.05, 1.20, 1.31, 1.55, 1.65]

NK_DIR = "./dataset/data1"
OUT_DIR = "./dataset/multicavity_smalltoken_gpu"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_RATIO = 0.8
SPLIT_SEED = 42
GLOBAL_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128 if torch.cuda.is_available() else 32

COMPLEX_DTYPE = torch.complex64
REAL_DTYPE = torch.float32


# =========================
# 1) 材料（你给的池）
# =========================
LOW_N  = ["MgF2", "SiO2", "ZnO"]
MID_N  = ["MgO", "Si3N4"]
HIGH_N = ["HfO2", "TiO2", "Ta2O5", "AlN", "Nb2O5", "ZnS", "ZnSe"]
UHIGH_N = ["Si", "a-Si"]
ABSORB  = ["ITO", "GaSb", "Ge"]
METAL   = ["Ag", "Au", "Al", "Al2O3", "TiN"]  # 本脚本做全介质多腔；若你要 metal-hybrid 也能加


# =========================
# 2) 业内常用 DBR 材料对
# =========================
INDUSTRY_DBR_PAIRS: List[Tuple[str, str]] = [
    ("TiO2", "SiO2"),
    ("Ta2O5", "SiO2"),
    ("HfO2", "SiO2"),
    ("Nb2O5", "SiO2"),
    ("Si3N4", "SiO2"),
    ("AlN", "SiO2"),
]

# 端镜 DBR（高反）
END_PAIR_MIN = 6
END_PAIR_MAX = 10

# 中间耦合镜（弱 DBR）
MID_PAIR_MIN = 1
MID_PAIR_MAX = 4

# 腔阶次（半波：t = m * λ0/(2n)）
CAVITY_ORDER_SET = [1, 2, 3]

# 腔材料（常见：SiO2/Si3N4/MgF2/MgO/ITO）
CAVITY_MATS = ["SiO2", "Si3N4", "MgF2", "MgO", "ITO"]

# 多腔数量及采样权重
CAVITY_COUNT_WEIGHTS = {
    2: 0.7,   # 2-cavity
    3: 0.3,   # 3-cavity
}


# =========================
# 3) nk 加载 -> torch
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
# 4) 离散厚度表：QW / HW(m阶)
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
            t_nm = lam0 * 1000.0 / (4.0 * n0)
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
                t_nm = (m * lam0 * 1000.0) / (2.0 * n0)
                thk_table[mat][(int(round(lam0 * 1000)), int(m))] = int(round(t_nm))
    return thk_table


# =========================
# 5) 生成多腔结构
# =========================
def sample_num_cavities() -> int:
    ks = list(CAVITY_COUNT_WEIGHTS.keys())
    ps = np.array([CAVITY_COUNT_WEIGHTS[k] for k in ks], dtype=np.float64)
    ps = ps / ps.sum()
    return int(np.random.choice(ks, p=ps).item())


def append_dbr(mats: List[str], thks: List[int],
               H: str, L: str, pairs: int, lambda0_nm: int,
               qw_table: Dict[str, Dict[int, int]],
               start_with_H: bool = True):
    # (H L)^pairs  or (L H)^pairs
    for i in range(pairs * 2):
        if start_with_H:
            mat = H if (i % 2 == 0) else L
        else:
            mat = L if (i % 2 == 0) else H
        mats.append(mat)
        thks.append(qw_table[mat][lambda0_nm])


def generate_multicavity(qw_table, hw_table) -> Tuple[List[str], List[int], dict]:
    """
    结构（N腔）：
      EndDBR + cavity1 + MidDBR + cavity2 + MidDBR + ... + cavityN + EndDBR

    约定：
      - EndDBR：高反，pairs_end ∈ [END_PAIR_MIN, END_PAIR_MAX]
      - MidDBR：耦合镜，pairs_mid ∈ [MID_PAIR_MIN, MID_PAIR_MAX]
      - DBR 材料对：业内常用 INDUSTRY_DBR_PAIRS
      - cavity：材料从 CAVITY_MATS，厚度 half-wave(m阶)
    """
    H, L = random.choice(INDUSTRY_DBR_PAIRS)

    lambda0_um = random.choice(LAMBDA0_SET_UM)
    lambda0_nm = int(round(lambda0_um * 1000))

    n_cav = sample_num_cavities()
    pairs_end = random.randint(END_PAIR_MIN, END_PAIR_MAX)
    pairs_mid = random.randint(MID_PAIR_MIN, MID_PAIR_MAX)

    # cavity 配置：每个腔可随机材料/阶次（更真实）
    cavity_cfg = []
    for _ in range(n_cav):
        cav_mat = random.choice(CAVITY_MATS)
        cav_order = random.choice(CAVITY_ORDER_SET)
        cav_thk = hw_table[cav_mat][(lambda0_nm, cav_order)]
        cavity_cfg.append((cav_mat, cav_order, cav_thk))

    mats: List[str] = []
    thks: List[int] = []

    # 左端镜（常用：以 H 开头）
    append_dbr(mats, thks, H, L, pairs_end, lambda0_nm, qw_table, start_with_H=True)

    # cavity + (mid mirror + cavity)...
    for ci in range(n_cav):
        cav_mat, cav_order, cav_thk = cavity_cfg[ci]
        mats.append(cav_mat)
        thks.append(int(cav_thk))

        if ci != n_cav - 1:
            # 中间耦合镜：通常与端镜同对，但 pairs 更小
            # 可选择翻转起始层以增强多样性
            startH = bool(random.getrandbits(1))
            append_dbr(mats, thks, H, L, pairs_mid, lambda0_nm, qw_table, start_with_H=startH)

    # 右端镜（常用：镜像，以 L 开头）
    append_dbr(mats, thks, H, L, pairs_end, lambda0_nm, qw_table, start_with_H=False)

    meta = dict(
        family="multicavity",
        mirror_pair=f"{H}/{L}",
        lambda0_um=float(lambda0_um),
        lambda0_nm=int(lambda0_nm),
        num_cavities=int(n_cav),
        end_pairs=int(pairs_end),
        mid_pairs=int(pairs_mid),
        cavities=[dict(mat=m, order=int(o), thk_nm=int(t)) for (m, o, t) in cavity_cfg],
        num_layers=int(len(mats)),
    )
    return mats, thks, meta


# =========================
# 6) batch 打包：pad 到 batch 内最大层数
# =========================
def pack_batch_to_tmm_inputs(batch_mats: List[List[str]],
                             batch_thks_nm: List[List[int]],
                             nk_dict_torch: Dict[str, torch.Tensor],
                             wl_m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回:
      n: [B, Lmax+2, num_wl] complex
      d: [B, Lmax+2] real (meters, with inf at ends)
    约定：入射/出射介质都为空气(1.0)
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
            n_pad_val = n[bi, L, :].clone()
            n[bi, 1 + L:1 + Lmax, :] = n_pad_val

    return n, d


# =========================
# 7) tmm_fast batch 计算
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
# 8) 保存 train/dev
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
# 9) 主流程
# =========================
def main():
    print("DEVICE =", DEVICE, "| BATCH_SIZE =", BATCH_SIZE)

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_SEED)

    # 需要的材料集合（DBR 对 + cavity mats）
    needed = set()
    for H, L in INDUSTRY_DBR_PAIRS:
        needed.add(H); needed.add(L)
    for m in CAVITY_MATS:
        needed.add(m)

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

    pbar = tqdm(total=NUM_SAMPLES, desc="Generating Multi-Cavity dataset (tmm_fast batch)")
    max_layers_seen = 0

    while len(struct_list) < NUM_SAMPLES:
        cur = min(BATCH_SIZE, NUM_SAMPLES - len(struct_list))
        batch_mats, batch_thks, batch_meta = [], [], []

        for _ in range(cur):
            mats, thks, meta = generate_multicavity(qw_table, hw_table)
            batch_mats.append(mats)
            batch_thks.append(thks)
            batch_meta.append(meta)
            max_layers_seen = max(max_layers_seen, meta["num_layers"])

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

    # ========== 统计：family / token / 最大层数 ==========
    tok_cnt = Counter()
    for seq in struct_list:
        for tok in seq:
            tok_cnt[tok] += 1

    print("\n== Max layers seen in generated data ==")
    print("max_layers_seen =", max_layers_seen, "| suggest max_len >= max_layers_seen + 2 (BOS/EOS)")

    print("\n== Top tokens (raw occurrences) ==")
    tot_tok = sum(tok_cnt.values())
    for t, c in tok_cnt.most_common(20):
        print(f"  {t:18s} {c:7d} ({c/tot_tok:.3%})")

    # ---- 理论 token space（按离散厚度规则推导）----
    possible_thk = defaultdict(set)

    # DBR 用到的材料：从 INDUSTRY_DBR_PAIRS 收集
    dbr_mats = set()
    for H, L in INDUSTRY_DBR_PAIRS:
        dbr_mats.add(H); dbr_mats.add(L)
    for mat in dbr_mats:
        for _, t in qw_table[mat].items():
            possible_thk[mat].add(int(t))

    # cavity 材料：half-wave(m阶)
    for mat in CAVITY_MATS:
        for _, t in hw_table[mat].items():
            possible_thk[mat].add(int(t))

    total_vocab = sum(len(v) for v in possible_thk.values())

    print("\n== Token space summary (theoretical by design) ==")
    for mat in sorted(possible_thk.keys()):
        thks = sorted(possible_thk[mat])
        k = len(thks)
        preview = thks[:10]
        print(f"  {mat:10s} unique_thk={k:3d}  thk_nm={preview}{'...' if k>10 else ''}")

    print("\n== TOTAL token categories (unique Material_ThicknessNm) ==")
    print("TOTAL_VOCAB =", total_vocab)
    print("OBSERVED_UNIQUE_TOKENS_IN_DATA =", len(tok_cnt))

    print("\nDone. OUT_DIR =", OUT_DIR)


if __name__ == "__main__":
    main()
