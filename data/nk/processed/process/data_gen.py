import numpy as np
import pandas as pd
import random
import os
from tqdm import tqdm
from scipy.interpolate import interp1d
from tmm import coh_tmm
import pickle as pkl
import matplotlib.pyplot as plt

# ============================================================
# 1. 全局配置
# ============================================================

NUM_SAMPLES = 6000

PAIR_MIN = 6
PAIR_MAX = 10

WAVELENGTHS = np.arange(0.8, 1.7, 0.005)  # um
LAMBDA0_RANGE = (0.9, 1.5)                # um

PERTURB_STD = 0.01
PERTURB_RANDOM = 0.1

DBR_MATERIAL_PAIRS = [
    ('TiO2', 'SiO2'),
    ('Ta2O5', 'SiO2'),
    ('HfO2', 'SiO2')
]

SUBSTRATE = 'Glass_Substrate'
SUBSTRATE_THICKNESS = 500000  # nm

NK_DIR = './data/nk/processed'
OUT_DIR = './output'
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 2. nk 数据加载
# ============================================================

def load_nk(materials, wavelengths):
    nk = {}
    for mat in materials:
        df = pd.read_csv(os.path.join(NK_DIR, mat + '.csv'))
        wl = df['wl'].values
        n = df['n'].values
        k = df['k'].values

        n_interp = interp1d(wl, n, bounds_error=False, fill_value='extrapolate')
        k_interp = interp1d(wl, k, bounds_error=False, fill_value='extrapolate')

        nk[mat] = n_interp(wavelengths) + 1j * k_interp(wavelengths)
    return nk

# ============================================================
# 3. DBR 结构生成
# ============================================================

def generate_dbr(nk_dict):
    H, L = random.choice(DBR_MATERIAL_PAIRS)
    lambda0 = random.uniform(*LAMBDA0_RANGE)
    pairs = random.randint(PAIR_MIN, PAIR_MAX)

    if random.random() < 0.3:
        perturb = PERTURB_STD
        mode = 'standard'
    else:
        perturb = PERTURB_RANDOM
        mode = 'random'

    materials = []
    thicknesses = []

    idx0 = np.argmin(np.abs(WAVELENGTHS - lambda0))

    for i in range(pairs * 2):
        mat = H if i % 2 == 0 else L
        n_real = np.real(nk_dict[mat][idx0])

        d = lambda0 * 1000 / (4 * n_real)
        d *= random.uniform(1 - perturb, 1 + perturb)

        materials.append(mat)
        thicknesses.append(d)

    return materials, thicknesses, lambda0, mode

# ============================================================
# 4. DBR 光谱计算（coh TMM） -> 同时输出 R, T
# ============================================================

def calc_RT(materials, thicknesses, nk_dict, pol='s', theta_deg=0.0):
    """
    Return:
        R_list, T_list (each length = len(WAVELENGTHS))
    """
    R, T = [], []
    d_list = [np.inf] + thicknesses + [np.inf]
    th0 = np.deg2rad(theta_deg)

    wl_list_nm = (WAVELENGTHS * 1000).astype(int)
    for i, wl_nm in enumerate(wl_list_nm):
        n_list = [1] + [nk_dict[m][i] for m in materials] + [1]
        res = coh_tmm(
            pol=pol,
            n_list=n_list,
            d_list=d_list,
            th_0=th0,
            lam_vac=wl_nm
        )
        R.append(res['R'])
        T.append(res['T'])

    return R, T

# ============================================================
# 5. 可视化（验证用）
# ============================================================

def visualize_one_dbr(nk_dict):
    mats, thks, lambda0, mode = generate_dbr(nk_dict)
    R, T = calc_RT(mats, thks, nk_dict)

    wl_nm = (WAVELENGTHS * 1000)

    plt.figure(figsize=(7, 4))
    plt.plot(wl_nm, R, lw=2, label='R')
    plt.plot(wl_nm, T, lw=2, label='T')
    plt.axvline(lambda0 * 1000, ls='--', c='r', label=f'λ₀ = {int(lambda0*1000)} nm')
    plt.ylim(0, 1.05)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Value')
    plt.title(f'DBR Spectrum | {mode} | {len(mats)//2} pairs')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Structure:")
    for m, t in zip(mats, thks):
        print(f"{m:6s}  {t:.1f} nm")

# ============================================================
# 6. 数据集生成（保存 R/T 两套列 + pkl 拼接 [R,T]）
# ============================================================

def generate_dataset():
    print("Loading nk data...")
    mats = list(set([m for p in DBR_MATERIAL_PAIRS for m in p]))
    nk = load_nk(mats, WAVELENGTHS)

    wl_nm_list = (WAVELENGTHS * 1000).astype(int)
    records = []
    dropped = 0

    for _ in tqdm(range(NUM_SAMPLES)):
        try:
            mats, thks, lambda0, mode = generate_dbr(nk)
            R, T = calc_RT(mats, thks, nk)

            rec = {
                'structure': [f'{m}_{int(round(t))}' for m, t in zip(mats, thks)],
                'num_layers': len(mats),
                'lambda0_nm': int(lambda0 * 1000),
                'mode': mode
            }

            # 写入列
            for wl, r in zip(wl_nm_list, R):
                rec[f'R_{wl}nm'] = r
            for wl, t in zip(wl_nm_list, T):
                rec[f'T_{wl}nm'] = t

            records.append(rec)

        except Exception:
            dropped += 1
            continue

    df = pd.DataFrame(records)

    df.to_csv(os.path.join(OUT_DIR, 'dbr_dataset.csv'), index=False)
    df.to_pickle(os.path.join(OUT_DIR, 'dbr_dataset.pkl'))

    # 结构 pkl（list[list[str]]）
    with open(os.path.join(OUT_DIR, 'total_structure.pkl'), 'wb') as f:
        pkl.dump(df['structure'].tolist(), f)

    # 光谱 pkl：每条样本拼接 [R..., T...]
    R_cols = [c for c in df.columns if c.startswith('R_')]
    T_cols = [c for c in df.columns if c.startswith('T_')]

    # 保证列顺序按波长升序（非常重要！）
    def wl_key(colname):
        # 'R_800nm' -> 800
        return int(colname.split('_')[1].replace('nm', ''))

    R_cols = sorted(R_cols, key=wl_key)
    T_cols = sorted(T_cols, key=wl_key)

    spec_mat = df[R_cols + T_cols].values.astype(np.float32)  # (N, 2*Nλ)
    with open(os.path.join(OUT_DIR, 'total_spectrum.pkl'), 'wb') as f:
        pkl.dump(spec_mat.tolist(), f)

    print(f"✔ DBR dataset generated: {len(df)} samples (dropped {dropped})")
    print(f"✔ spectrum dim = {spec_mat.shape[1]} (= 2 * {len(R_cols)})")
    print(f"Saved to: {OUT_DIR}")

# ============================================================
# 7. 主入口
# ============================================================

if __name__ == '__main__':
    mats = list(set([m for p in DBR_MATERIAL_PAIRS for m in p]))
    nk = load_nk(mats, WAVELENGTHS)

    # 先画几条 sanity check
    # for _ in range(3):
    #     visualize_one_dbr(nk)

    generate_dataset()
