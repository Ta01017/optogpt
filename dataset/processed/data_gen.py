
import os
import random
import pickle as pkl
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from tmm import coh_tmm
import matplotlib.pyplot as plt

# =========================
# 0) 配置区
# =========================

NUM_SAMPLES = 60000

PAIR_MIN = 6
PAIR_MAX = 10

WAVELENGTHS = np.linspace(0.9, 1.7, int(round((1.7 - 0.9) / 0.005)) + 1)  # um  # um

LAMBDA0_RANGE = (0.9, 1.7)                # um

PERTURB_STD = 0.01
PERTURB_RANDOM = 0.10
P_MODE_STD_PROB = 0.30

NK_DIR = './dataset/data'
PAIR_CSV = './dataset/processed/dbr_exploratory_pairs.csv'

OUT_DIR = './pkl'
os.makedirs(OUT_DIR, exist_ok=True)

# 训练用输出目录
DATASET_DIR = './dataset/dbr'
os.makedirs(DATASET_DIR, exist_ok=True)

# 划分比例：train/dev = 8/2
TRAIN_RATIO = 0.8
SPLIT_SEED = 42  # 保证可复现划分

SANITY_PLOT_NUM = 0  # 0关闭

# ===== 验证策略 =====
VERIFY_MODE = 'all'         # 'all' or 'sample'
SPECTRUM_CHECK_PROB = 0.05  # VERIFY_MODE='sample' 时抽检比例
MAX_TRIES = NUM_SAMPLES * 30

# ===== 你要的：只验证业内常用pair =====
ONLY_INDUSTRY = True   # True: 只生成业内常用pair；False: 按原分层策略采样

# =========================
# 1) 业内常用白名单
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
# 2) 分层阈值与采样比例
# =========================
K_MAX_LIMIT = 5e-4
CROSS_RATE_MAX = 0.0

DN_STRONG = 0.8
DN_MID = 0.35
DN_WEAK = 0.2

SAMPLING_WEIGHTS = dict(
    industry=0.50,
    strong=0.25,
    mid=0.20,
    weak=0.05
)

# =========================
# 3) DBR 验证阈值
# =========================
MAX_REL_ERR_QW = 0.25

PEAK_MIN = 0.85
BAND_THRESHOLD = 0.80
BAND_MIN_WIDTH_NM = 50


# =========================
# 工具：加载 nk
# =========================
def load_nk(materials, wavelengths):
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

        nk[mat] = n_interp(wavelengths) + 1j * k_interp(wavelengths)
    return nk


# =========================
# 工具：读取 pair CSV + 分层
# =========================
@dataclass(frozen=True)
class PairInfo:
    H: str
    L: str
    dn_med: float
    cross_rate: float
    k_max: float
    source: str
    tier: str


def load_and_bucket_pairs(pair_csv: str):
    df = pd.read_csv(pair_csv)
    df = df[(df['k_max'] <= K_MAX_LIMIT) & (df['cross_rate'] <= CROSS_RATE_MAX)].copy()

    industry_set = set(INDUSTRY_CORE)
    buckets = dict(industry=[], strong=[], mid=[], weak=[])

    # ---- 先从CSV里收集 industry ----
    for _, r in df.iterrows():
        H, L = str(r['H']), str(r['L'])
        dn = float(r['dn_med'])
        cr = float(r['cross_rate'])
        km = float(r['k_max'])

        if (H, L) in industry_set:
            buckets['industry'].append(PairInfo(H, L, dn, cr, km, source='industry', tier='industry'))
            continue

        # 若不是 ONLY_INDUSTRY 才填其它tier
        if not ONLY_INDUSTRY:
            if dn >= DN_STRONG:
                tier = 'strong'
            elif dn >= DN_MID:
                tier = 'mid'
            elif dn >= DN_WEAK:
                tier = 'weak'
            else:
                continue
            buckets[tier].append(PairInfo(H, L, dn, cr, km, source='explore', tier=tier))

    # ---- 补齐 industry 白名单（CSV 没有也保留）----
    existing_industry = set((p.H, p.L) for p in buckets['industry'])
    for (H, L) in INDUSTRY_CORE:
        if (H, L) not in existing_industry:
            buckets['industry'].append(
                PairInfo(H, L, dn_med=-1.0, cross_rate=0.0, k_max=-1.0, source='industry', tier='industry')
            )

    # ---- ONLY_INDUSTRY模式：强制清空其它bucket ----
    if ONLY_INDUSTRY:
        buckets['strong'].clear()
        buckets['mid'].clear()
        buckets['weak'].clear()

    print("== Pair buckets ==")
    for k, v in buckets.items():
        print(f"{k:8s}: {len(v)}")

    if ONLY_INDUSTRY and len(buckets['industry']) == 0:
        raise RuntimeError("ONLY_INDUSTRY=True but industry bucket is empty. Check INDUSTRY_CORE or CSV.")

    return buckets


def sample_pair(buckets):
    if ONLY_INDUSTRY:
        if len(buckets['industry']) == 0:
            raise RuntimeError("industry bucket empty under ONLY_INDUSTRY=True")
        return random.choice(buckets['industry'])

    keys = list(SAMPLING_WEIGHTS.keys())
    weights = [SAMPLING_WEIGHTS[k] for k in keys]
    tier = random.choices(keys, weights=weights, k=1)[0]

    non_empty = [k for k in keys if len(buckets[k]) > 0]
    if tier not in non_empty:
        tier = random.choice(non_empty)

    return random.choice(buckets[tier])


# =========================
# DBR 结构生成
# =========================
def generate_dbr(nk_dict, buckets):
    pair = sample_pair(buckets)
    H, L = pair.H, pair.L

    lambda0 = random.uniform(*LAMBDA0_RANGE)
    pairs = random.randint(PAIR_MIN, PAIR_MAX)

    if random.random() < P_MODE_STD_PROB:
        perturb = PERTURB_STD
        mode = 'standard'
    else:
        perturb = PERTURB_RANDOM
        mode = 'random'

    idx0 = int(np.argmin(np.abs(WAVELENGTHS - lambda0)))

    materials, thicknesses = [], []
    for i in range(pairs * 2):
        mat = H if (i % 2 == 0) else L
        n_real = float(np.real(nk_dict[mat][idx0]))
        d = lambda0 * 1000.0 / (4.0 * n_real)  # nm
        d *= random.uniform(1 - perturb, 1 + perturb)
        materials.append(mat)
        thicknesses.append(d)

    meta = dict(
        H=H, L=L,
        dn_med=pair.dn_med,
        pair_tier=pair.tier,
        pair_source=pair.source,
        lambda0_um=lambda0,
        lambda0_nm=int(lambda0 * 1000),
        pairs=pairs,
        mode=mode,
        perturb=perturb
    )
    return materials, thicknesses, meta


# =========================
# 光谱计算：R/T
# =========================
def calc_RT(materials, thicknesses, nk_dict, pol='s', theta_deg=0.0):
    R, T = [], []
    d_list = [np.inf] + thicknesses + [np.inf]
    th0 = np.deg2rad(theta_deg)
    # wl_nm_list = (WAVELENGTHS * 1000).astype(int)
    wl_nm_list = np.round(WAVELENGTHS * 1000).astype(int)

    for i, wl_nm in enumerate(wl_nm_list):
        n_list = [1] + [nk_dict[m][i] for m in materials] + [1]
        res = coh_tmm(pol=pol, n_list=n_list, d_list=d_list, th_0=th0, lam_vac=wl_nm)
        R.append(res['R'])
        T.append(res['T'])
    return np.asarray(R, dtype=np.float32), np.asarray(T, dtype=np.float32)


# =========================
# 结构语法检查（快）
# =========================
def check_dbr_structure(materials, thicknesses, nk_dict, lambda0_um, expected_pair=None,
                        max_rel_err=MAX_REL_ERR_QW, require_alternating=True):
    if len(materials) != len(thicknesses):
        return False, {"reason": "len(materials)!=len(thicknesses)"}

    if require_alternating and len(materials) >= 2:
        for i in range(1, len(materials)):
            if materials[i] == materials[i - 1]:
                return False, {"reason": "not alternating", "pos": i}

    if expected_pair is not None:
        H, L = expected_pair
        for i, m in enumerate(materials):
            if (i % 2 == 0 and m != H) or (i % 2 == 1 and m != L):
                return False, {"reason": "pair mismatch", "expected": f"{H}/{L}"}

    idx0 = int(np.argmin(np.abs(WAVELENGTHS - lambda0_um)))
    rel_errs = []
    for m, d in zip(materials, thicknesses):
        n0 = float(np.real(nk_dict[m][idx0]))
        d_qw = lambda0_um * 1000.0 / (4.0 * n0)
        rel_errs.append(abs(d - d_qw) / max(d_qw, 1e-9))

    rel_errs = np.asarray(rel_errs, dtype=np.float32)
    info = {
        "qw_rel_err_mean": float(rel_errs.mean()),
        "qw_rel_err_max": float(rel_errs.max())
    }
    if info["qw_rel_err_max"] > max_rel_err:
        return False, {"reason": "quarter-wave deviation too large", **info}

    return True, info


# =========================
# 光谱行为检查（慢）
# =========================
def check_dbr_spectrum(R, lambda0_um, peak_min=PEAK_MIN, band_threshold=BAND_THRESHOLD, band_min_width_nm=BAND_MIN_WIDTH_NM):
    wl_nm = np.round(WAVELENGTHS * 1000).astype(int)
    lambda0_nm = lambda0_um * 1000.0
    R = np.asarray(R, dtype=np.float32)

    win = (wl_nm >= (lambda0_nm - 50)) & (wl_nm <= (lambda0_nm + 50))
    if win.sum() < 5:
        win = (wl_nm >= (lambda0_nm - 80)) & (wl_nm <= (lambda0_nm + 80))

    peak = float(R[win].max())
    if peak < peak_min:
        return False, {"reason": "peak too low", "peak": peak}

    mask = R >= band_threshold
    idx0 = int(np.argmin(np.abs(wl_nm - lambda0_nm)))
    if not mask[idx0]:
        return False, {"reason": "lambda0 not inside high-R band", "peak": peak, "band_threshold": band_threshold}

    l = idx0
    while l > 0 and mask[l - 1]:
        l -= 1
    r = idx0
    while r < len(mask) - 1 and mask[r + 1]:
        r += 1

    width_nm = float(wl_nm[r] - wl_nm[l])
    metrics = {"peak": peak, "band_width_nm": width_nm, "band_threshold": band_threshold}

    if width_nm < band_min_width_nm:
        return False, {"reason": "stop-band too narrow", **metrics}

    return True, metrics


# =========================
# 保存 total + split
# =========================
def save_total_and_split(df: pd.DataFrame):
    # ---- total 保存（不划分）----
    total_csv = os.path.join(OUT_DIR, 'dbr_total_dataset.csv')
    total_pkl = os.path.join(OUT_DIR, 'dbr_total_dataset.pkl')
    df.to_csv(total_csv, index=False)
    df.to_pickle(total_pkl)

    # total pkl（结构 / 光谱）
    with open(os.path.join(OUT_DIR, 'total_structure.pkl'), 'wb') as f:
        pkl.dump(df['structure'].tolist(), f)

    # spectrum: [R..., T...]
    R_cols = [c for c in df.columns if c.startswith('R_')]
    T_cols = [c for c in df.columns if c.startswith('T_')]

    def wl_key(colname):
        return int(colname.split('_')[1].replace('nm', ''))

    R_cols = sorted(R_cols, key=wl_key)
    T_cols = sorted(T_cols, key=wl_key)
    spec_mat = df[R_cols + T_cols].values.astype(np.float32)

    with open(os.path.join(OUT_DIR, 'total_spectrum.pkl'), 'wb') as f:
        pkl.dump(spec_mat.tolist(), f)

    # ---- split train/dev ----
    N = len(df)
    rng = np.random.default_rng(SPLIT_SEED)
    idx = rng.permutation(N)

    split = int(N * TRAIN_RATIO)
    idx_train = idx[:split]
    idx_dev = idx[split:]

    # 保证对应关系：结构和光谱用同样索引
    total_struct = df['structure'].tolist()
    total_spec = spec_mat.tolist()

    train_struct = [total_struct[i] for i in idx_train]
    train_spec = [total_spec[i] for i in idx_train]

    dev_struct = [total_struct[i] for i in idx_dev]
    dev_spec = [total_spec[i] for i in idx_dev]

    with open(os.path.join(DATASET_DIR, 'Structure_train.pkl'), 'wb') as f:
        pkl.dump(train_struct, f)
    with open(os.path.join(DATASET_DIR, 'Spectrum_train.pkl'), 'wb') as f:
        pkl.dump(train_spec, f)
    with open(os.path.join(DATASET_DIR, 'Structure_dev.pkl'), 'wb') as f:
        pkl.dump(dev_struct, f)
    with open(os.path.join(DATASET_DIR, 'Spectrum_dev.pkl'), 'wb') as f:
        pkl.dump(dev_spec, f)

    # 小报告
    print("\n== Saved total ==")
    print("  -", total_csv)
    print("  -", total_pkl)
    print("  -", os.path.join(OUT_DIR, 'total_structure.pkl'))
    print("  -", os.path.join(OUT_DIR, 'total_spectrum.pkl'))

    print("\n== Saved split train/dev ==")
    print(f"  Train: {len(train_struct)} | Dev: {len(dev_struct)} | Ratio={TRAIN_RATIO}")
    print("  -", os.path.join(DATASET_DIR, 'Structure_train.pkl'))
    print("  -", os.path.join(DATASET_DIR, 'Spectrum_train.pkl'))
    print("  -", os.path.join(DATASET_DIR, 'Structure_dev.pkl'))
    print("  -", os.path.join(DATASET_DIR, 'Structure_dev.pkl'))
    print("  -", os.path.join(DATASET_DIR, 'Spectrum_dev.pkl'))

    print("\nNote: train_spectrum 每条为 [R..., T...]，训练时 spec_type=R/T/R_T 决定读取方式。")


# =========================
# 主生成
# =========================
def generate_dataset():
    buckets = load_and_bucket_pairs(PAIR_CSV)

    # ONLY_INDUSTRY模式：只检查白名单材料是否都有nk文件
    if ONLY_INDUSTRY:
        needed = set()
        for (H, L) in INDUSTRY_CORE:
            needed.add(H)
            needed.add(L)
        missing = [m for m in sorted(needed) if not os.path.exists(os.path.join(NK_DIR, f"{m}.csv"))]
        if missing:
            raise FileNotFoundError(
                "Missing nk csv for industry materials:\n  " + "\n  ".join(missing) +
                f"\nExpected under: {NK_DIR}"
            )

    mats_needed = set()
    for tier, plist in buckets.items():
        for p in plist:
            mats_needed.add(p.H)
            mats_needed.add(p.L)

    print("Loading nk...")
    nk_dict = load_nk(sorted(mats_needed), WAVELENGTHS)

    # sanity plot（可选）
    if SANITY_PLOT_NUM > 0:
        wl_nm = WAVELENGTHS * 1000
        for _ in range(SANITY_PLOT_NUM):
            mats, thks, meta = generate_dbr(nk_dict, buckets)
            ok_s, info_s = check_dbr_structure(mats, thks, nk_dict, meta['lambda0_um'], expected_pair=(meta['H'], meta['L']))
            R, T = calc_RT(mats, thks, nk_dict)
            ok_r, info_r = check_dbr_spectrum(R, meta['lambda0_um'])
            plt.figure(figsize=(8, 4))
            plt.plot(wl_nm, R, lw=2, label='R')
            plt.plot(wl_nm, T, lw=2, label='T')
            plt.axvline(meta['lambda0_nm'], ls='--', c='r', label=f"λ0={meta['lambda0_nm']}nm")
            plt.ylim(0, 1.05)
            plt.title(f"sanity | {meta['pair_tier']} | {meta['H']}/{meta['L']} | ok={ok_s and ok_r}")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()
            print("structure_ok:", ok_s, info_s, "spec_ok:", ok_r, info_r)

    wl_nm_list = np.round(WAVELENGTHS * 1000).astype(int)
    records = []
    dropped = 0
    tried = 0
    fail_struct = 0
    fail_spec = 0

    pbar = tqdm(total=NUM_SAMPLES, desc="Generating DBR (verified)")

    while len(records) < NUM_SAMPLES and tried < MAX_TRIES:
        tried += 1
        try:
            mats, thks, meta = generate_dbr(nk_dict, buckets)

            ok_s, info_s = check_dbr_structure(
                mats, thks, nk_dict, meta['lambda0_um'],
                expected_pair=(meta['H'], meta['L'])
            )
            if not ok_s:
                fail_struct += 1
                continue

            R, T = calc_RT(mats, thks, nk_dict)

            do_spec_check = (VERIFY_MODE == 'all') or (random.random() < SPECTRUM_CHECK_PROB)
            if do_spec_check:
                ok_r, info_r = check_dbr_spectrum(R, meta['lambda0_um'])
                if not ok_r:
                    fail_spec += 1
                    continue
            else:
                ok_r, info_r = True, {"skipped": True}

            rec = {
                'structure': [f'{m}_{int(round(t))}' for m, t in zip(mats, thks)],
                'num_layers': len(mats),
                'pairs': meta['pairs'],
                'lambda0_nm': meta['lambda0_nm'],
                'mode': meta['mode'],
                'perturb': meta['perturb'],
                'pair_H': meta['H'],
                'pair_L': meta['L'],
                'pair_name': f"{meta['H']}/{meta['L']}",
                'pair_tier': meta['pair_tier'],
                'pair_source': meta['pair_source'],
                'dn_med': meta['dn_med'],
                'qw_rel_err_mean': info_s.get('qw_rel_err_mean', np.nan),
                'qw_rel_err_max': info_s.get('qw_rel_err_max', np.nan),
                'spec_check_ok': bool(ok_r),
                'spec_check_peak': info_r.get('peak', np.nan),
                'spec_check_band_width_nm': info_r.get('band_width_nm', np.nan),
                'spec_check_threshold': info_r.get('band_threshold', np.nan),
                'spec_check_skipped': bool(info_r.get('skipped', False)),
            }

            for wl, r in zip(wl_nm_list, R):
                rec[f'R_{wl}nm'] = float(r)
            for wl, t in zip(wl_nm_list, T):
                rec[f'T_{wl}nm'] = float(t)

            records.append(rec)
            pbar.update(1)

        except Exception:
            dropped += 1
            continue

    pbar.close()

    if len(records) < NUM_SAMPLES:
        print(f"\n[WARN] Only generated {len(records)}/{NUM_SAMPLES} after {tried} tries.")
        print("Consider loosening thresholds (PEAK_MIN/BAND_MIN_WIDTH_NM/MAX_REL_ERR_QW) or increasing MAX_TRIES.")

    df = pd.DataFrame(records)

    print("\n== Tier distribution ==")
    print(df['pair_tier'].value_counts(normalize=True).sort_index())

    print("\n== Generation report ==")
    print(f"Generated: {len(df)} | Tried: {tried} | Dropped(except checks): {dropped}")
    print(f"Fail(struct): {fail_struct} | Fail(spec): {fail_spec}")
    print(f"Wavelength points: {len(WAVELENGTHS)} | Spectrum dim: {2*len(WAVELENGTHS)}")

    # 统一保存（total + split）
    save_total_and_split(df)


if __name__ == '__main__':
    generate_dataset()
