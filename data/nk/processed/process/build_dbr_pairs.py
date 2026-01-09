import os, glob
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

NK_DIR = '..'
WAVELENGTHS = np.arange(0.8, 1.7, 0.005)  # um

# 过滤阈值（你可以按材料实际质量调）
K_THRESH = 1e-2       # 探索性先放宽点，工业常用可用 1e-3
DN_THRESH = 0.2       # 折射率反差下限
CROSS_MAX = 0.05      # 允许最多 5% 波长点出现 n_H <= n_L
N_MIN, N_MAX = 1.1, 4.0

def load_one(mat_path):
    df = pd.read_csv(mat_path).dropna()
    wl = df['wl'].values
    n = df['n'].values
    k = df['k'].values
    n_fn = interp1d(wl, n, bounds_error=False, fill_value='extrapolate')
    k_fn = interp1d(wl, k, bounds_error=False, fill_value='extrapolate')
    n_i = n_fn(WAVELENGTHS)
    k_i = k_fn(WAVELENGTHS)
    return n_i, k_i

# 1) 读取材料
materials = {}
for p in glob.glob(os.path.join(NK_DIR, '*.csv')):
    name = os.path.splitext(os.path.basename(p))[0]
    n_i, k_i = load_one(p)
    materials[name] = (n_i.astype(np.float32), k_i.astype(np.float32))

# 2) 计算材料统计并过滤
stats = {}
cands = []
for name, (n_i, k_i) in materials.items():
    n_med = float(np.median(n_i))
    k_max = float(np.max(k_i))
    n_rng = float(np.max(n_i) - np.min(n_i))
    stats[name] = dict(n_med=n_med, k_max=k_max, n_rng=n_rng)

    if (N_MIN <= n_med <= N_MAX) and (k_max <= K_THRESH):
        cands.append(name)

# 3) 枚举材料对并筛选 DBR 合法 pairs
pairs = []
for i in range(len(cands)):
    for j in range(i+1, len(cands)):
        a, b = cands[i], cands[j]
        na, ka = materials[a]
        nb, kb = materials[b]

        # 设定 H/L
        if stats[a]['n_med'] >= stats[b]['n_med']:
            H, L = a, b
            nH, nL = na, nb
            kH, kL = ka, kb
        else:
            H, L = b, a
            nH, nL = nb, na
            kH, kL = kb, ka

        dn = float(np.median(nH - nL))
        cross_rate = float(np.mean((nH - nL) <= 0))
        k_pair_max = float(max(np.max(kH), np.max(kL)))

        if dn >= DN_THRESH and cross_rate <= CROSS_MAX and k_pair_max <= K_THRESH:
            pairs.append((H, L, dn, cross_rate, k_pair_max))

# 4) 按 dn 排序保存
pairs = sorted(pairs, key=lambda x: x[2], reverse=True)

out = pd.DataFrame(pairs, columns=['H', 'L', 'dn_med', 'cross_rate', 'k_max'])
out.to_csv('dbr_exploratory_pairs.csv', index=False)

print(f"Total materials: {len(materials)}")
print(f"Candidate low-loss materials: {len(cands)}")
print(f"Exploratory DBR pairs found: {len(pairs)}")
print("Top-10 pairs:")
print(out.head(10))
