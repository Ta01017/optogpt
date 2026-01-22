#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import numpy as np
import pandas as pd


def _mae_backsample_check(
    wl_excel: np.ndarray,   # [Nw_excel]
    T: np.ndarray,          # [K, Nw_excel]  (原始曲线，已做数值化/单位处理)
    wl95: np.ndarray,       # [95]
    W_raw: np.ndarray,      # [K,95] 插值后的曲线
):
    """
    反采样一致性检测：
    - 把插值后的 W_raw 再采样回 Excel 原始的 wl（仅在 wl95 范围内）
    - 与 Excel 原始曲线中间段对比，输出 MAE / MAX
    """
    wl_min, wl_max = float(wl95.min()), float(wl95.max())
    mask_mid = (wl_excel >= wl_min) & (wl_excel <= wl_max)

    wl_mid = wl_excel[mask_mid]
    if wl_mid.size < 3:
        print("[Warn] wl95 覆盖的区间在 wl_excel 上点太少，无法做可靠 back-sample 检测。")
        return None

    maes, maxes = [], []
    for k in range(T.shape[0]):
        t_mid = T[k, mask_mid]
        # 从 wl95 上的插值曲线反采样回 wl_mid
        t_back = np.interp(wl_mid, wl95, W_raw[k]).astype(np.float32)

        # 如果原曲线里有 NaN，这里要忽略
        valid = np.isfinite(t_mid) & np.isfinite(t_back)
        if valid.sum() == 0:
            maes.append(np.nan)
            maxes.append(np.nan)
            continue

        diff = np.abs(t_back[valid] - t_mid[valid])
        maes.append(float(diff.mean()))
        maxes.append(float(diff.max()))

    maes = np.array(maes, dtype=np.float32)
    maxes = np.array(maxes, dtype=np.float32)

    print("\n[Interp Check] Back-sample consistency on wl95 range")
    print(f"  wl95 range: {wl_min:.3f} ~ {wl_max:.3f} (len={len(wl95)})")
    print(f"  wl_mid len: {len(wl_mid)} (points from excel within wl95 range)")
    print("  per-curve MAE :", np.round(maes, 6))
    print("  per-curve MAX :", np.round(maxes, 6))
    print(f"  mean(MAE)={float(np.nanmean(maes)):.6f} | mean(MAX)={float(np.nanmean(maxes)):.6f}")
    print(f"  worst MAE={float(np.nanmax(maes)):.6f} | worst MAX={float(np.nanmax(maxes)):.6f}")

    return maes, maxes


def build_W_from_excel(
    xlsx_path: str,
    out_npz: str = "W_6x95_from_excel_T.npz",
    wl0: float = 900.0,
    wl1: float = 1670.0,
    target_bands_115: int = 115,
    crop_left: int = 15,
    crop_right: int = 5,
    normalize: str = "sum1",   # "sum1" or None
    assume_percent_if_max_gt: float = 2.0,  # 如果 T.max > 2，自动认为是百分比(0-100)并/100
):
    df = pd.read_excel(xlsx_path)

    # 1) 抽取所有 T_***nm 列，并解析波长
    pat = re.compile(r"^T_(\d+)nm$")
    t_cols, wl_excel = [], []
    for c in df.columns:
        m = pat.match(str(c))
        if m:
            t_cols.append(c)
            wl_excel.append(int(m.group(1)))

    if len(t_cols) == 0:
        raise ValueError("没找到任何列名形如 T_900nm 的列，请检查 Excel 列名。")

    wl_excel = np.array(wl_excel, dtype=np.float32)
    order = np.argsort(wl_excel)
    wl_excel = wl_excel[order]
    t_cols = [t_cols[i] for i in order]

    # 2) 取出 K 行曲线数据: [K, Nw]
    #    注意：用 to_numeric 强制数值化，避免 Unnamed/空字符串导致 NaN 扩散
    T = df[t_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    K, Nw = T.shape

    print(f"[Info] Loaded T curves: K={K}, Nw={Nw}, wl_excel range=({wl_excel.min()}, {wl_excel.max()})")
    nan_ratio = float(np.isnan(T).mean())
    print(f"[Info] T nan ratio: {nan_ratio:.6f}")

    # 2.1) 单位检查：是否是百分比(0~100)
    tmax = float(np.nanmax(T))
    print(f"[Info] T.max = {tmax:.6f}")
    if tmax > assume_percent_if_max_gt:
        print(f"[Warn] T 看起来像百分比(0~100)，自动执行 T/=100（阈值 {assume_percent_if_max_gt}）")
        T = T / 100.0

    # 3) 构造你代码对应的 115→95 波长网格
    wl115 = np.linspace(wl0, wl1, target_bands_115, dtype=np.float32)  # 115
    wl95  = wl115[crop_left: target_bands_115 - crop_right]            # 95
    expected_95 = target_bands_115 - crop_left - crop_right
    assert wl95.shape[0] == expected_95, f"wl95 len mismatch: got {len(wl95)} expected {expected_95}"

    # 4) 插值到 wl95：得到 W_raw [K,95]
    #    left/right：这里延用你原逻辑（边界用首末值延拓），物理上更稳
    W_raw = np.stack(
        [np.interp(wl95, wl_excel, T[k], left=T[k, 0], right=T[k, -1]) for k in range(K)],
        axis=0
    ).astype(np.float32)

    # 4.1) 新增：插值一致性检测（反采样）
    _mae_backsample_check(wl_excel, T, wl95, W_raw)

    # 5) 行归一化（推荐，保证输出尺度稳定）
    if normalize == "sum1":
        W = W_raw / (W_raw.sum(axis=1, keepdims=True) + 1e-8)
    else:
        W = W_raw

    np.savez(out_npz, W=W, W_raw=W_raw, wl_excel=wl_excel, wl95=wl95, wl115=wl115)
    print(f"\n[OK] Saved to {out_npz}")
    print("W shape:", W.shape, "wl95 head/tail:", wl95[:3], wl95[-3:])
    print("\n[Diag] Row sums (W_raw):", np.round(W_raw.sum(axis=1), 4))
    print("[Diag] Row sums (W):    ", np.round(W.sum(axis=1), 4))
    print("[Diag] Peak wl (W_raw):", [float(wl95[np.argmax(W_raw[k])]) for k in range(K)])

    return W, wl95


if __name__ == "__main__":
    xlsx_path = "mentor_curves.xlsx"  # TODO: 改成你的路径
    build_W_from_excel(xlsx_path)
