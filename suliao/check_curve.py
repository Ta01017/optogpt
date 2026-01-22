#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_modulation_curves_ABC.py

一次性完成你要的 A/B/C 三个检查，并给出结论提示：
A) 覆盖与扎堆：每条曲线的中心波长、有效带宽、覆盖区间
B) 冗余：相关系数矩阵 + SVD 有效秩/能量占比
C) 与“均匀 6 个带通”对比：两种定义方式（按 wl95 / 按 band index）

输入：
- 你之前保存的 npz：包含 W_raw, W, wl95
  例如 W_6x95_from_excel_T.npz

输出：
- 打印分析报告
- 保存图：
  - abc_1_curves_raw_vs_uniform.png
  - abc_2_corr_heatmap.png
  - abc_3_svd_energy.png
  - abc_4_bandpass_defs_compare.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def weighted_center(wl, w):
    s = w.sum() + 1e-12
    return float((wl * w).sum() / s)


def effective_bandwidth(wl, w, frac=0.2):
    """返回超过 frac*max 的连续覆盖区间（可能多段），以及总宽度（nm）。"""
    thr = frac * float(w.max())
    mask = w >= thr
    if not mask.any():
        return [], 0.0
    idx = np.where(mask)[0]
    # 找连续段
    segments = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            segments.append((start, prev))
            start = i
            prev = i
    segments.append((start, prev))

    seg_nm = [(float(wl[a]), float(wl[b])) for a, b in segments]
    total_bw = sum(float(wl[b] - wl[a]) for a, b in segments)
    return seg_nm, total_bw


def make_uniform_bandpass_by_index(C=95, K=6):
    """
    均匀带通（按 band index 均分）：
    - 不关心真实波长，只把 95 个通道平均分为 6 段，每段权重相等。
    - 若 95 不能整除 6，会出现边界段略宽/略窄（用“近似等长切分”处理）。
    """
    W = np.zeros((K, C), dtype=np.float32)
    # 等比例切分边界（使每段包含的通道数尽量均匀）
    edges = np.linspace(0, C, K + 1)
    edges = np.round(edges).astype(int)
    edges[0] = 0
    edges[-1] = C
    for k in range(K):
        a, b = edges[k], edges[k + 1]
        if b <= a:
            b = min(C, a + 1)
        W[k, a:b] = 1.0
        W[k] /= (W[k].sum() + 1e-12)  # 行归一化
    return W


def make_uniform_bandpass_by_wavelength(wl, K=6):
    """
    均匀带通（按 wl95 均分“波长跨度”）：
    - 按波长轴把范围均分为 6 段，然后把落在每段里的 band index 赋权为 1。
    - 若 wl 不是等间隔，这种定义和“按 index 均分”会有明显不同。
    """
    wl = np.asarray(wl, dtype=np.float32)
    C = wl.shape[0]
    W = np.zeros((K, C), dtype=np.float32)
    wl0, wl1 = float(wl.min()), float(wl.max())
    bounds = np.linspace(wl0, wl1, K + 1)

    for k in range(K):
        lo, hi = bounds[k], bounds[k + 1]
        # 最后一段包含上边界
        if k < K - 1:
            mask = (wl >= lo) & (wl < hi)
        else:
            mask = (wl >= lo) & (wl <= hi)
        if not mask.any():
            # 兜底：挑最近的一个 band
            idx = int(np.argmin(np.abs(wl - (lo + hi) / 2)))
            mask[idx] = True
        W[k, mask] = 1.0
        W[k] /= (W[k].sum() + 1e-12)
    return W


def corrcoef_rows(W):
    """计算行相关系数矩阵（KxK）。"""
    # 防止全零/常数导致 nan
    Wc = W - W.mean(axis=1, keepdims=True)
    denom = (np.linalg.norm(Wc, axis=1, keepdims=True) + 1e-12)
    Wn = Wc / denom
    return (Wn @ Wn.T).clip(-1, 1)


def svd_energy(W):
    """返回奇异值与能量占比（按 s^2）。"""
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    e = s ** 2
    ratio = e / (e.sum() + 1e-12)
    cumsum = np.cumsum(ratio)
    return s, ratio, cumsum


def analyze_and_plot(npz_path: str, out_dir: str = ".", frac_bw: float = 0.2):
    os.makedirs(out_dir, exist_ok=True)
    z = np.load(npz_path)
    # 你的 npz key 是大写 W/W_raw
    W = z["W"]
    W_raw = z["W_raw"]
    wl = z["wl95"]

    K, C = W.shape
    assert W_raw.shape == (K, C)
    assert wl.shape[0] == C

    print(f"\n[Load] {npz_path}")
    print(f"  W shape     : {W.shape} (normalized)")
    print(f"  W_raw shape : {W_raw.shape} (interpolated, not normalized)")
    print(f"  wl95 range  : {float(wl.min()):.3f} ~ {float(wl.max()):.3f} (len={len(wl)})")

    # ===== A) 覆盖/扎堆 =====
    print("\n=== A) Coverage / Bandwidth / Center ===")
    centers = []
    bws = []
    segs_all = []
    for k in range(K):
        c0 = weighted_center(wl, W_raw[k])
        segs, bw_nm = effective_bandwidth(wl, W_raw[k], frac=frac_bw)
        centers.append(c0)
        bws.append(bw_nm)
        segs_all.append(segs)
        print(f"curve{k}: center={c0:.2f}nm | bw@{int(frac_bw*100)}%max={bw_nm:.2f}nm | segments={segs}")

    centers = np.array(centers, dtype=np.float32)
    bws = np.array(bws, dtype=np.float32)

    # 简单“扎堆”判断：中心波长的跨度
    span = float(centers.max() - centers.min())
    print(f"\n[A-Insight] centers span = {span:.2f} nm (越小越扎堆)")
    if span < 0.35 * (float(wl.max() - wl.min())):
        print("[A-Insight] ⚠️ 6条曲线的中心波长跨度偏小，可能扎堆在局部波段，覆盖不均衡。")
    else:
        print("[A-Insight] ✅ 中心波长覆盖跨度尚可（仅供粗判断，仍需看图与带宽）。")

    # ===== B) 冗余：相关性 + SVD =====
    print("\n=== B) Redundancy: Corr + SVD ===")
    corr = corrcoef_rows(W_raw)
    print("[B] corr matrix (rows):\n", np.round(corr, 3))

    # 冗余指标：平均绝对非对角相关
    off = corr[~np.eye(K, dtype=bool)]
    mean_abs_off = float(np.mean(np.abs(off)))
    print(f"[B-Insight] mean |corr(off-diag)| = {mean_abs_off:.3f} (越接近1越冗余)")
    if mean_abs_off > 0.85:
        print("[B-Insight] ⚠️ 通道高度相关（冗余大），6维里有效信息维度可能明显<6。")
    elif mean_abs_off > 0.65:
        print("[B-Insight] ⚠️ 通道相关性偏高，可能比“均匀带通”更不互补。")
    else:
        print("[B-Insight] ✅ 相关性不算太夸张（仍建议结合SVD看有效秩）。")

    s, ratio, cumsum = svd_energy(W_raw)
    print("[B] singular values:", np.round(s, 4))
    print("[B] energy ratio   :", np.round(ratio, 4))
    print("[B] energy cumsum  :", np.round(cumsum, 4))
    eff_rank_90 = int(np.searchsorted(cumsum, 0.90) + 1)
    eff_rank_95 = int(np.searchsorted(cumsum, 0.95) + 1)
    print(f"[B-Insight] effective rank: 90%->{eff_rank_90} dims, 95%->{eff_rank_95} dims (越小越“少数维度主导”)")

    # ===== C) 与均匀带通对比（两种定义） =====
    W_u_idx = make_uniform_bandpass_by_index(C=C, K=K)
    W_u_wl = make_uniform_bandpass_by_wavelength(wl, K=K)

    # “按 wl 均分”和“按 index 均分”差异量化
    diff = np.mean(np.abs(W_u_idx - W_u_wl))
    print("\n=== C) Uniform bandpass comparison ===")
    print(f"[C] mean |W_uniform_index - W_uniform_wl| = {diff:.6f}")
    if diff < 1e-6:
        print("[C-Insight] ✅ 你这里 wl95 近似等间隔，所以两种均匀带通几乎等价。")
    else:
        print("[C-Insight] ⚠️ 两种均匀带通不等价，说明 wl95 非等间隔（或边界划分导致段落包含band数不同）。")

    # ===== 画图 =====

    # 图1：W_raw vs 两种均匀带通（叠加）
    plt.figure()
    for k in range(K):
        plt.plot(wl, W_raw[k], label=f"excel_raw curve{k}")
    for k in range(K):
        plt.plot(wl, W_u_idx[k] * W_raw.max(), linestyle="--", label=f"uniform_by_index{k} (scaled)")
    for k in range(K):
        plt.plot(wl, W_u_wl[k] * W_raw.max(), linestyle=":", label=f"uniform_by_wl{k} (scaled)")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Value")
    plt.title("Excel curves (W_raw) vs Uniform bandpass (scaled for visibility)")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    p1 = os.path.join(out_dir, "abc_1_curves_raw_vs_uniform.png")
    plt.savefig(p1, dpi=220)

    # 图2：相关系数热力图
    plt.figure()
    plt.imshow(corr, vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(K), [f"c{k}" for k in range(K)])
    plt.yticks(range(K), [f"c{k}" for k in range(K)])
    plt.title("Row correlation of W_raw (redundancy check)")
    plt.tight_layout()
    p2 = os.path.join(out_dir, "abc_2_corr_heatmap.png")
    plt.savefig(p2, dpi=220)

    # 图3：SVD能量占比
    plt.figure()
    x = np.arange(1, len(ratio) + 1)
    plt.plot(x, ratio, marker="o", label="energy ratio (s^2)")
    plt.plot(x, cumsum, marker="o", label="cumulative")
    plt.xlabel("Component")
    plt.ylabel("Energy")
    plt.title("SVD energy of W_raw (effective rank)")
    plt.legend()
    plt.tight_layout()
    p3 = os.path.join(out_dir, "abc_3_svd_energy.png")
    plt.savefig(p3, dpi=220)

    # 图4：两种均匀带通的具体边界差异（可视化）
    plt.figure()
    for k in range(K):
        plt.plot(wl, W_u_idx[k], label=f"uniform_index{k}")
    for k in range(K):
        plt.plot(wl, W_u_wl[k], linestyle="--", label=f"uniform_wl{k}")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Weight (row-normalized)")
    plt.title("Uniform bandpass definition: by index vs by wavelength")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    p4 = os.path.join(out_dir, "abc_4_bandpass_defs_compare.png")
    plt.savefig(p4, dpi=220)

    plt.show()

    # ===== 给出“为什么可能比均匀带通还差”的自动化提示 =====
    print("\n=== Auto diagnosis summary ===")
    # 1) 覆盖扎堆
    total_range = float(wl.max() - wl.min())
    if span < 0.35 * total_range:
        print("- 可能原因①：6条曲线中心波长扎堆（覆盖不均），导致某些关键波段被忽略。")
    # 2) 冗余
    if mean_abs_off > 0.75 or eff_rank_90 <= 3:
        print("- 可能原因②：曲线高度相关/有效秩低（等效通道数<6），信息互补性不如均匀带通。")
    # 3) 带宽过宽或过窄
    bw_med = float(np.median(bws))
    if bw_med > 0.7 * total_range:
        print("- 可能原因③：曲线过宽（接近全谱平滑加权），相当于做低频平均，判别信息被抹平。")
    if np.any(bws < 0.05 * total_range):
        print("- 可能原因④：有曲线过窄或接近“点采样”，对噪声更敏感，且与相机band对齐误差更致命。")
    print("- 建议：若要用这些物理曲线但又想保住分类性能，考虑：W固定 + 后接可学习1×1 conv补偿，或直接让W可微调。")
    print(f"\n[Saved figs]\n  {p1}\n  {p2}\n  {p3}\n  {p4}\n")


if __name__ == "__main__":
    # TODO: 改成你的 npz 文件路径
    NPZ_PATH = "W_6x95_from_excel_T.npz"
    OUT_DIR = "abc_out"
    analyze_and_plot(NPZ_PATH, out_dir=OUT_DIR, frac_bw=0.2)
