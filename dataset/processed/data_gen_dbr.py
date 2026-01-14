#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_dbr_small_token_30k.py

目标：
- 生成 DBR 数据集（Structure/Spectrum），但让结构 token 数量很小：
  * 不使用随机厚度扰动
  * 厚度只来源于少量中心波长 lambda0 的 quarter-wave 厚度（四分之一波厚度）
  * token 仍然为 "Material_ThicknessNm"，但每种材料只有 ~len(LAMBDA0_SET) 种厚度

输出：
OUT_DIR/
  Structure_train.pkl
  Spectrum_train.pkl
  Structure_dev.pkl
  Spectrum_dev.pkl
  meta_list.pkl  (每条样本的 pair_name、lambda0_nm、pairs、num_layers 等)

注意：
- Spectrum 每条为 [R..., T...]，波长网格与你原来一致：0.9~1.7 um, step=0.005
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
from tmm import coh_tmm


# =========================
# 0) 配置
# =========================
NUM_SAMPLES = 30000

PAIR_MIN = 6
PAIR_MAX = 10

# 波长网格（保持与你原代码一致）
WAVELENGTHS_UM = np.linspace(0.9, 1.7, int(round((1.7 - 0.9) / 0.005)) + 1)

# 关键：只允许少量中心波长 lambda0（决定厚度 token 数量）
# 覆盖 0.9~1.7，并包含通信常用的 1.31/1.55（只是常见波段点）
LAMBDA0_SET_UM = [0.95, 1.05, 1.20, 1.31, 1.55, 1.65]  # 6个点 -> 每种材料最多6种厚度

NK_DIR = './dataset/data'
OUT_DIR = './dataset/dbr_smalltoken'
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_RATIO = 0.8
SPLIT_SEED = 42

# 为了可复现
GLOBAL_SEED = 42


# =========================
# 1) 业内常用材料对（你已有）
#    （常见 DBR 材料体系中，SiO2 作为低折材料很常见，配 TiO2/Ta2O5/HfO2/Nb2O5 等高折）
# =========================
INDUSTRY_CORE: List[Tuple[str, str]] = [
    ('TiO2', 'SiO2'),
    ('Ta2O5', 'SiO2'),
    ('HfO2', 'SiO2'),
    ('Nb2O5', 'SiO2'),
    ('Si3N4', 'SiO2'),
    ('AlN', 'SiO2'),
]


# =========================
# 2) 工具：加载 nk（wl,n,k csv -> 插值到 WAVELENGTHS_UM）
# =========================
def load_nk(materials: List[str], wavelengths_um: np.ndarray) -> Dict[str, np.ndarray]:
    nk = {}
    for mat in materials:
        path = os.path.join(NK_DIR, f'{mat}.csv')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing nk csv for material: {mat} | expected: {path}")

        df = pd.read_csv(path).dropna()
        wl = df['wl'].values
        n = df['n'].values
        k = df['k'].values

        n_interp = interp1d(wl, n, bounds_error=False, fill_value='extrapolate')
        k_interp = interp1d(wl, k, bounds_error=False, fill_value='extrapolate')

        nk[mat] = n_interp(wavelengths_um) + 1j * k_interp(wavelengths_um)
    return nk


# =========================
# 3) 预计算离散厚度表：每个材料在每个 lambda0 下的 quarter-wave 厚度
#    t = lambda0/(4*n(lambda0))  （四分之一波厚度）
# =========================
def precompute_qw_thickness_table(nk_dict: Dict[str, np.ndarray],
                                  wavelengths_um: np.ndarray,
                                  lambda0_set_um: List[float]) -> Dict[str, Dict[int, int]]:
    wl = wavelengths_um
    thk_table: Dict[str, Dict[int, int]] = {}
    for mat, nk_arr in nk_dict.items():
        thk_table[mat] = {}
        for lam0 in lambda0_set_um:
            idx0 = int(np.argmin(np.abs(wl - lam0)))
            n0 = float(np.real(nk_arr[idx0]))
            t_nm = lam0 * 1000.0 / (4.0 * n0)      # nm
            t_nm_int = int(round(t_nm))            # 让 token 更干净稳定
            thk_table[mat][int(round(lam0 * 1000))] = t_nm_int
    return thk_table


# =========================
# 4) 光谱计算：R/T
# =========================
def calc_RT(materials: List[str],
            thicknesses_nm: List[int],
            nk_dict: Dict[str, np.ndarray],
            wavelengths_um: np.ndarray,
            pol: str = 's',
            theta_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    R, T = [], []
    d_list = [np.inf] + list(thicknesses_nm) + [np.inf]
    th0 = np.deg2rad(theta_deg)
    wl_nm_list = np.round(wavelengths_um * 1000).astype(int)

    for i, wl_nm in enumerate(wl_nm_list):
        n_list = [1] + [nk_dict[m][i] for m in materials] + [1]
        res = coh_tmm(pol=pol, n_list=n_list, d_list=d_list, th_0=th0, lam_vac=int(wl_nm))
        R.append(res['R'])
        T.append(res['T'])

    return np.asarray(R, np.float32), np.asarray(T, np.float32)


# =========================
# 5) 生成 DBR（离散厚度）
# =========================
def generate_dbr_discrete(industry_pairs: List[Tuple[str, str]],
                          thk_table: Dict[str, Dict[int, int]]) -> Tuple[List[str], List[int], dict]:
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
        mode="discrete_qw"
    )
    return materials, thicknesses, meta


# =========================
# 6) 保存 train/dev 的 pkl
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

    with open(os.path.join(out_dir, 'Structure_train.pkl'), 'wb') as f:
        pkl.dump(train_struct, f)
    with open(os.path.join(out_dir, 'Spectrum_train.pkl'), 'wb') as f:
        pkl.dump(train_spec, f)
    with open(os.path.join(out_dir, 'Structure_dev.pkl'), 'wb') as f:
        pkl.dump(dev_struct, f)
    with open(os.path.join(out_dir, 'Spectrum_dev.pkl'), 'wb') as f:
        pkl.dump(dev_spec, f)

    with open(os.path.join(out_dir, 'meta_train.pkl'), 'wb') as f:
        pkl.dump(train_meta, f)
    with open(os.path.join(out_dir, 'meta_dev.pkl'), 'wb') as f:
        pkl.dump(dev_meta, f)

    print("\n== Saved ==")
    print("Train:", len(train_struct), "Dev:", len(dev_struct))
    print(" -", os.path.join(out_dir, 'Structure_train.pkl'))
    print(" -", os.path.join(out_dir, 'Spectrum_train.pkl'))
    print(" -", os.path.join(out_dir, 'Structure_dev.pkl'))
    print(" -", os.path.join(out_dir, 'Spectrum_dev.pkl'))
    print(" -", os.path.join(out_dir, 'meta_train.pkl'))
    print(" -", os.path.join(out_dir, 'meta_dev.pkl'))


# =========================
# 7) 主流程
# =========================
def main():
    # seeds
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    # 检查 nk 文件
    needed = set()
    for H, L in INDUSTRY_CORE:
        needed.add(H)
        needed.add(L)

    missing = [m for m in sorted(needed) if not os.path.exists(os.path.join(NK_DIR, f"{m}.csv"))]
    if missing:
        raise FileNotFoundError(
            "Missing nk csv:\n  " + "\n  ".join(missing) +
            f"\nExpected under: {NK_DIR}"
        )

    # 加载 nk
    nk_dict = load_nk(sorted(list(needed)), WAVELENGTHS_UM)

    # 预计算离散厚度表
    thk_table = precompute_qw_thickness_table(nk_dict, WAVELENGTHS_UM, LAMBDA0_SET_UM)

    # 打印：每材料厚度 token 数量（应≈len(LAMBDA0_SET_UM) 或更少(若有重复 round)）
    print("== Discrete thickness table (nm) ==")
    for m in sorted(thk_table.keys()):
        vals = sorted(set(thk_table[m].values()))
        print(f"{m:8s} unique_thk={len(vals):2d}  values={vals}")

    # 生成
    struct_list = []
    spec_list = []
    meta_list = []

    pbar = tqdm(total=NUM_SAMPLES, desc="Generating DBR small-token 30k")
    while len(struct_list) < NUM_SAMPLES:
        mats, thks, meta = generate_dbr_discrete(INDUSTRY_CORE, thk_table)
        R, T = calc_RT(mats, thks, nk_dict, WAVELENGTHS_UM)

        struct_tokens = [f"{m}_{int(t)}" for m, t in zip(mats, thks)]
        spec_vec = np.concatenate([R, T], axis=0).astype(np.float32).tolist()

        struct_list.append(struct_tokens)
        spec_list.append(spec_vec)
        meta_list.append(meta)
        pbar.update(1)
    pbar.close()

    # 保存 train/dev
    save_split(struct_list, spec_list, meta_list, OUT_DIR)

    # =========================
    # 统计：pair 分布、(pair, lambda0) 覆盖、token top
    # =========================
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

    # 额外：估算 token 空间规模（粗略）
    mats = sorted(list(needed))
    approx_unique_thk = {m: len(set(thk_table[m].values())) for m in mats}
    approx_vocab = sum(approx_unique_thk.values())  # 近似：每个材料的厚度token数相加
    print("\n== Approx token space ==")
    print("materials:", mats)
    print("unique_thk per material:", approx_unique_thk)
    print("approx total structure tokens (material_thk):", approx_vocab)
    print("(+ BOS/EOS/PAD/UNK 之后仍是几十级别)")

    print("\nDone. OUT_DIR =", OUT_DIR)


if __name__ == "__main__":
    main()
