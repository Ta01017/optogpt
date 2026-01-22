#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_excel_T(xlsx_path: str):
    df = pd.read_excel(xlsx_path)

    pat = re.compile(r"^T_(\d+)nm$")
    t_cols, wl = [], []
    for c in df.columns:
        m = pat.match(str(c))
        if m:
            t_cols.append(c)
            wl.append(int(m.group(1)))

    if not t_cols:
        raise ValueError("没找到任何 T_900nm 这类列，请检查列名。")

    wl = np.array(wl, dtype=np.float32)
    order = np.argsort(wl)
    wl = wl[order]
    t_cols = [t_cols[i] for i in order]

    T = df[t_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)  # [K, Nw]
    return wl, T

def build_wl95(wl0=900.0, wl1=1670.0, n115=115, crop_left=15, crop_right=5):
    wl115 = np.linspace(wl0, wl1, n115, dtype=np.float32)
    wl95  = wl115[crop_left: n115 - crop_right]
    return wl95

def main(xlsx_path="mentor_curves.xlsx", curve_id=0):
    wl_excel, T = load_excel_T(xlsx_path)
    wl95 = build_wl95()

    print("[Info] wl_excel:", wl_excel.min(), wl_excel.max(), "len", len(wl_excel))
    print("[Info] wl95    :", wl95.min(), wl95.max(), "len", len(wl95))
    print("[Info] T shape :", T.shape)
    print("[Info] T nan ratio:", float(np.isnan(T).mean()))

    # 1) 单位检查：是不是百分比
    tmax = float(np.nanmax(T))
    print("[Info] T.max =", tmax)
    if tmax > 2.0:
        print("[Warn] T 看起来像百分比(0-100)，建议 /100 后再用")
        T_use = T / 100.0
    else:
        T_use = T

    # 2) 取一条曲线，先把原曲线裁到 wl95 范围，避免“区间不一致”导致的错觉
    k = curve_id
    mask_mid = (wl_excel >= wl95.min()) & (wl_excel <= wl95.max())
    wl_mid = wl_excel[mask_mid]
    t_mid  = T_use[k, mask_mid]

    # 3) 插值到 wl95
    t_interp = np.interp(wl95, wl_excel, T_use[k], left=np.nan, right=np.nan)

    # 4) 再反向验证：把插值结果采样回 wl_mid，看与原始中段点差多少（这一步能判断插值是否真的“错了”）
    t_back = np.interp(wl_mid, wl95, t_interp)
    mae = float(np.nanmean(np.abs(t_back - t_mid)))
    mx  = float(np.nanmax(np.abs(t_back - t_mid)))
    print(f"[Check] mid-range back-sample MAE={mae:.6f}, MAX={mx:.6f}")

    # 5) 画图：只在同一波段区间比较
    plt.figure()
    plt.plot(wl_excel, T_use[k], label=f"Excel raw curve{k}")
    plt.plot(wl95, t_interp, "--", label="Interpolated on wl95")
    plt.xlim(float(wl95.min()), float(wl95.max()))
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("T")
    plt.title("Compare on the SAME wavelength range (wl95)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
