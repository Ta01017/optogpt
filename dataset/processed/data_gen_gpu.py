# generate_dbr_60k_gpu.py
import os
import random
import pickle as pkl
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d

import torch
from tmm_fast import coh_tmm  # ✅ GPU + 向量化

# =========================
# 0) 配置区
# =========================

NUM_SAMPLES = 60000
PAIR_MIN = 6
PAIR_MAX = 10

WAVELENGTHS = np.linspace(0.9, 1.7, int(round((1.7 - 0.9) / 0.005)) + 1)  # um
WL_NM = torch.tensor(np.round(WAVELENGTHS * 1000).astype(np.float32), device="cuda")

LAMBDA0_RANGE = (0.9, 1.7)

PERTURB_STD = 0.01
PERTURB_RANDOM = 0.10
P_MODE_STD_PROB = 0.30

NK_DIR = './dataset/data'
PAIR_CSV = './dataset/processed/dbr_exploratory_pairs.csv'

OUT_DIR = './pkl'
DATASET_DIR = './dataset/dbr'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

TRAIN_RATIO = 0.8
SPLIT_SEED = 42

# ===== GPU 友好验证策略 =====
VERIFY_MODE = 'sample'
SPECTRUM_CHECK_PROB = 0.02

PEAK_MIN = 0.65
BAND_THRESHOLD = 0.60
BAND_MIN_WIDTH_NM = 20

MAX_TRIES = NUM_SAMPLES * 20

ONLY_INDUSTRY = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 1) 业内常用 DBR pair
# =========================

INDUSTRY_CORE = [
    ('TiO2', 'SiO2'),
    ('Ta2O5', 'SiO2'),
    ('HfO2', 'SiO2'),
    ('Nb2O5', 'SiO2'),
    ('Si3N4', 'SiO2'),
    ('AlN', 'SiO2'),
]

# =========================
# nk 加载
# =========================

def load_nk(materials):
    nk = {}
    for mat in materials:
        df = pd.read_csv(os.path.join(NK_DIR, f'{mat}.csv')).dropna()
        n_interp = interp1d(df['wl'], df['n'], fill_value='extrapolate')
        k_interp = interp1d(df['wl'], df['k'], fill_value='extrapolate')
        nk_complex = n_interp(WAVELENGTHS) + 1j * k_interp(WAVELENGTHS)
        nk[mat] = torch.tensor(nk_complex, dtype=torch.complex64, device=DEVICE)
    return nk

# =========================
# Pair 读取
# =========================

@dataclass(frozen=True)
class PairInfo:
    H: str
    L: str

def load_pairs():
    pairs = []
    for H, L in INDUSTRY_CORE:
        pairs.append(PairInfo(H, L))
    return pairs

# =========================
# DBR 生成
# =========================

def generate_dbr(pairs, nk_dict):
    pair = random.choice(pairs)
    H, L = pair.H, pair.L

    lambda0 = random.uniform(*LAMBDA0_RANGE)
    pairs_n = random.randint(PAIR_MIN, PAIR_MAX)

    perturb = PERTURB_STD if random.random() < P_MODE_STD_PROB else PERTURB_RANDOM

    idx0 = int(np.argmin(np.abs(WAVELENGTHS - lambda0)))

    mats, thks = [], []
    for i in range(pairs_n * 2):
        mat = H if i % 2 == 0 else L
        n0 = nk_dict[mat][idx0].real.item()
        d = lambda0 * 1000 / (4 * n0)
        d *= random.uniform(1 - perturb, 1 + perturb)
        mats.append(mat)
        thks.append(d)

    return mats, thks, lambda0

# =========================
# GPU RT 计算
# =========================


import torch
import numpy as np
from tmm_fast.vectorized_tmm_dispersive_multistack import coh_vec_tmm_disp_mstack as coh_tmm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# WAVELENGTHS: um -> nm, shape (W,)
WL_NM = torch.from_numpy(np.round(WAVELENGTHS * 1000).astype(np.float32)).to(DEVICE)  # (W,)


print("WL_NM shape:", WL_NM.shape)
print("lambda_vacuum shape:", WL_NM.view(-1).shape)

def calc_RT_gpu(mats, thks, nk_dict, pol='s', theta_deg=0.0):
    # ---- lambda_vacuum: MUST be 1D (W,)
    lambda_vacuum = WL_NM.view(-1)  # ✅ (W,)

    # ---- Theta: multistack usually expects (S,) where S=#stacks
    theta = torch.tensor([np.deg2rad(theta_deg)], device=DEVICE, dtype=torch.float32)  # (1,)

    # ---- thickness T: (S, L)
    T = torch.tensor([[float("inf")] + list(map(float, thks)) + [float("inf")]],
                     device=DEVICE, dtype=torch.float32)  # (1, L)

    # ---- refractive index N: (S, L, W)
    W = lambda_vacuum.shape[0]
    onesW = torch.ones(W, device=DEVICE, dtype=torch.complex64)

    layers = [onesW]  # incident
    for m in mats:
        # nk_dict[m] 需要是 torch.complex64 且 shape=(W,)
        n_w = nk_dict[m]
        if not torch.is_tensor(n_w):
            raise TypeError("nk_dict[m] must be torch.Tensor on GPU")
        n_w = n_w.to(device=DEVICE, dtype=torch.complex64).view(-1)
        assert n_w.shape[0] == W, f"nk length mismatch: {m}: {n_w.shape[0]} vs W={W}"
        layers.append(n_w)
    layers.append(onesW)  # exit

    N = torch.stack(layers, dim=0).unsqueeze(0)  # (1, L, W)

    # ✅ 顺序： (pol, N, T, lambda_vacuum, Theta)
    res = coh_tmm(pol, N, T, lambda_vacuum, theta)

    R = res["R"][0].detach().cpu().numpy().astype(np.float32)  # (W,)
    Tout = res["T"][0].detach().cpu().numpy().astype(np.float32)
    return R, Tout


# =========================
# 主生成流程
# =========================

def generate_dataset():
    pairs = load_pairs()
    nk_dict = load_nk({p.H for p in pairs} | {p.L for p in pairs})

    records = []
    tried = 0

    pbar = tqdm(total=NUM_SAMPLES, desc="Generating DBR (GPU)")

    while len(records) < NUM_SAMPLES and tried < MAX_TRIES:
        tried += 1

        mats, thks, lambda0 = generate_dbr(pairs, nk_dict)

        R, T = calc_RT_gpu(mats, thks, nk_dict)

        if VERIFY_MODE == 'sample' and random.random() < SPECTRUM_CHECK_PROB:
            if not check_dbr_spectrum_gpu(R, lambda0):
                continue

        rec = {
            'structure': [f'{m}_{int(round(t))}' for m, t in zip(mats, thks)]
        }
        for wl, r in zip(np.round(WAVELENGTHS * 1000).astype(int), R):
            rec[f'R_{wl}nm'] = float(r)
        for wl, t in zip(np.round(WAVELENGTHS * 1000).astype(int), T):
            rec[f'T_{wl}nm'] = float(t)

        records.append(rec)
        pbar.update(1)

    pbar.close()

    if len(records) == 0:
        raise RuntimeError("❌ 0 samples generated — DBR checks still too strict.")

    df = pd.DataFrame(records)
    print(f"Generated {len(df)} samples")

    # 保存
    df.to_pickle(os.path.join(OUT_DIR, 'dbr_total_dataset.pkl'))

    structs = df['structure'].tolist()
    specs = df[[c for c in df.columns if c.startswith('R_') or c.startswith('T_')]].values.tolist()

    idx = np.random.permutation(len(df))
    split = int(len(df) * TRAIN_RATIO)

    with open(os.path.join(DATASET_DIR, 'Structure_train.pkl'), 'wb') as f:
        pkl.dump([structs[i] for i in idx[:split]], f)
    with open(os.path.join(DATASET_DIR, 'Spectrum_train.pkl'), 'wb') as f:
        pkl.dump([specs[i] for i in idx[:split]], f)
    with open(os.path.join(DATASET_DIR, 'Structure_dev.pkl'), 'wb') as f:
        pkl.dump([structs[i] for i in idx[split:]], f)
    with open(os.path.join(DATASET_DIR, 'Spectrum_dev.pkl'), 'wb') as f:
        pkl.dump([specs[i] for i in idx[split:]], f)

    print("✅ Dataset saved")

# =========================
if __name__ == "__main__":
    generate_dataset()
