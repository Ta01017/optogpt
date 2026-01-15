#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_gen_fp_maxlen22.py

【FP 小长度可学习版】
- 仅生成：
  (A) MIM
  (B) Hybrid FP: 单侧 DBR + cavity + metal
- 严格限制最大层数 <= 10（max_len<=22 绝对安全）
- token = "Material_ThicknessNm"
"""

import os, random, pickle as pkl
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d

import torch
import tmm_fast

# =========================
# 基本配置
# =========================
NUM_SAMPLES = 30000
TRAIN_RATIO = 0.8
GLOBAL_SEED = 42
SPLIT_SEED = 42

WAVELENGTHS_UM = np.linspace(0.9, 1.7, int(round((1.7 - 0.9)/0.005)) + 1)
LAMBDA0_SET_UM = [1.05, 1.31, 1.55]   # ↓↓↓ 进一步减 token

NK_DIR = "./dataset/data1"
OUT_DIR = "./dataset/fp_maxlen22"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128 if torch.cuda.is_available() else 32

COMPLEX_DTYPE = torch.complex64
REAL_DTYPE = torch.float32

# =========================
# 材料池（强收敛）
# =========================
DBR_PAIRS = [
    ("Ta2O5", "SiO2"),
    ("Si3N4", "SiO2"),
]

CAVITY_MATS = ["SiO2", "ITO"]
METALS = ["Ag", "Al"]

DBR_PAIR_MAX = 4      # ← 核心
METAL_THK = [20, 30, 40]
ITO_THK = [120, 200, 400]

FAMILY_WEIGHTS = {
    "hybrid": 0.7,
    "mim": 0.3,
}

# =========================
# nk 加载
# =========================
def load_nk(materials):
    nk = {}
    for m in materials:
        df = pd.read_csv(os.path.join(NK_DIR, f"{m}.csv")).dropna()
        wl = df["wl"].values
        n = interp1d(wl, df["n"].values, fill_value="extrapolate")(WAVELENGTHS_UM)
        k = interp1d(wl, df["k"].values, fill_value="extrapolate")(WAVELENGTHS_UM)
        nk[m] = torch.tensor(n + 1j*k, dtype=COMPLEX_DTYPE, device=DEVICE)
    return nk

# =========================
# quarter-wave DBR
# =========================
def qw_thk(nk, mat, lambda0_nm):
    idx = np.argmin(np.abs(WAVELENGTHS_UM*1000 - lambda0_nm))
    n0 = float(np.real(nk[mat][idx].cpu().numpy()))
    return int(round(lambda0_nm / (4*n0)))

# =========================
# 结构生成
# =========================
def gen_hybrid(nk):
    H, L = random.choice(DBR_PAIRS)
    pairs = random.randint(2, DBR_PAIR_MAX)
    lambda0_nm = int(round(random.choice(LAMBDA0_SET_UM)*1000))

    mats, thks = [], []
    for i in range(pairs*2):
        m = H if i%2==0 else L
        mats.append(m)
        thks.append(qw_thk(nk, m, lambda0_nm))

    cav = random.choice(CAVITY_MATS)
    mats.append(cav)
    thks.append(random.choice(ITO_THK if cav=="ITO" else [qw_thk(nk,cav,lambda0_nm)*2]))

    metal = random.choice(METALS)
    mats.append(metal)
    thks.append(random.choice(METAL_THK))

    return mats, thks, {"family":"hybrid","layers":len(mats)}

def gen_mim():
    metal = random.choice(METALS)
    return (
        [metal,"ITO",metal],
        [random.choice(METAL_THK), random.choice(ITO_THK), random.choice(METAL_THK)],
        {"family":"mim","layers":3}
    )

def sample_struct(nk):
    if random.random() < FAMILY_WEIGHTS["hybrid"]:
        return gen_hybrid(nk)
    return gen_mim()

# =========================
# tmm_fast batch
# =========================
def pack(batch_mats, batch_thks, nk, wl_m):
    B = len(batch_mats)
    Lmax = max(len(x) for x in batch_mats)
    W = wl_m.shape[0]

    d = torch.zeros((B,Lmax+2),device=DEVICE)
    d[:,0]=d[:,-1]=float("inf")

    n = torch.zeros((B,Lmax+2,W),dtype=COMPLEX_DTYPE,device=DEVICE)
    n[:,0,:]=1.0
    n[:,-1,:]=1.0

    for i,(mats,thks) in enumerate(zip(batch_mats,batch_thks)):
        d[i,1:1+len(mats)] = torch.tensor(thks,device=DEVICE)*1e-9
        for j,m in enumerate(mats):
            n[i,1+j,:]=nk[m]
    return n,d

def calc_spec(batch_mats,batch_thks,nk,wl_m):
    n,d=pack(batch_mats,batch_thks,nk,wl_m)
    out=tmm_fast.coh_tmm("s",n,d,torch.tensor([0.0],device=DEVICE),wl_m)
    R,T=out["R"][:,0,:],out["T"][:,0,:]
    return torch.cat([R,T],dim=1).cpu().numpy()

# =========================
# 主流程
# =========================
def main():
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    needed=set(sum(DBR_PAIRS,()))|set(CAVITY_MATS)|set(METALS)
    nk=load_nk(sorted(needed))

    wl_m=torch.tensor(WAVELENGTHS_UM*1e-6,device=DEVICE)

    structs,specs,metas=[],[],[]
    pbar=tqdm(total=NUM_SAMPLES)

    while len(structs)<NUM_SAMPLES:
        cur=min(BATCH_SIZE,NUM_SAMPLES-len(structs))
        bm,bt,bmeta=[],[],[]
        for _ in range(cur):
            m,t,meta=sample_struct(nk)
            bm.append(m);bt.append(t);bmeta.append(meta)

        spec=calc_spec(bm,bt,nk,wl_m)
        for i in range(cur):
            structs.append([f"{m}_{int(t)}" for m,t in zip(bm[i],bt[i])])
            specs.append(spec[i].tolist())
            metas.append(bmeta[i])
        pbar.update(cur)

    pbar.close()

    idx=np.random.permutation(len(structs))
    s=int(len(idx)*TRAIN_RATIO)
    for name,arr in [("Structure",structs),("Spectrum",specs),("meta",metas)]:
        pkl.dump([arr[i] for i in idx[:s]],open(f"{OUT_DIR}/{name}_train.pkl","wb"))
        pkl.dump([arr[i] for i in idx[s:]],open(f"{OUT_DIR}/{name}_dev.pkl","wb"))

    print("Done. OUT_DIR =",OUT_DIR)

if __name__=="__main__":
    main()
