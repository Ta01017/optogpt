#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_optogpt_universal_mae_tmmfast_full.py

通用 eval（tmm_fast 版，支持 batch，适配大规模混合数据集）：
- Oracle MAE: GT struct -> TMM_FAST -> spec_pred vs dev_spec
- Pred   MAE: greedy decode -> TMM_FAST -> spec_pred vs dev_spec
- 打印若干样例：type/num_layers + gt/pred pair + head + oracle/pred mae
- 输出整体 mae + by-type / by-num_layers bucket + pred pair/token 分布
- 【曲线差异验证】按 type / 层数 bucket 保存 GT/Oracle/Pred 的均值曲线 + 误差曲线（服务器可跑，不 show，只 save）
- 【新增】在保存的曲线图上写入“代表曲线差异的指标”（Oracle vs GT / Pred vs GT）：
    MAE / RMSE / MAX / Area|Δ| / Corr / PSNR
  并可选保存 mean_signed_err 图（看系统性偏差）

【重要：边界条件默认与生成器一致】
POL="s"
INC_MEDIUM="air"
EXIT_MEDIUM="substrate"
SUBSTRATE="Glass_Substrate"
FORCE_SUBSTRATE_K0=True  (substrate 半无限介质强制 k=0，避免 tmm_fast assert & 保持一致)

要求：
- token 格式：Material_ThicknessNm（例如 TiO2_145）
- thickness 单位：nm（脚本会 nm->m）
- nk csv: {mat}.csv with columns: wl,n,k (wl unit matches wavelength grid, usually um)
"""

import os
import json
import argparse
import random
import pickle as pkl
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from scipy.interpolate import interp1d

# ===== server-safe plotting (no display) =====
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tmm_fast
from core.models.transformer import make_model_I, subsequent_mask

# =========================
# Default physics settings (MATCH GENERATOR)
# =========================
POL = "s"
INC_MEDIUM = "air"
EXIT_MEDIUM = "substrate"          # "air" or "substrate"
SUBSTRATE = "Glass_Substrate"
FORCE_SUBSTRATE_K0 = True          # force k=0 for semi-infinite exit medium

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMPLEX_DTYPE = torch.complex64
REAL_DTYPE = torch.float32


# -------------------------
# IO / seed
# -------------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pkl.load(f)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------
# token parsing helpers
# -------------------------
def parse_structure_tokens(tokens: List[str]) -> Tuple[List[str], List[float]]:
    """Parse tokens like 'TiO2_145' into (mats, thks_nm). Ignore specials/unparsable."""
    mats, thks = [], []
    for s in tokens:
        if "_" not in s:
            continue
        m, t = s.split("_", 1)
        try:
            thk = float(t)
        except Exception:
            continue
        if thk < 0:
            continue
        mats.append(m)
        thks.append(thk)
    return mats, thks

def infer_pair_name(tokens: List[str]) -> str:
    """Just for human-readable printing: first two materials."""
    mats = []
    for s in tokens:
        if "_" not in s:
            continue
        mats.append(s.split("_", 1)[0])
        if len(mats) >= 2:
            break
    if len(mats) < 2:
        return "INVALID"
    return f"{mats[0]}/{mats[1]}"

def collect_materials_from_struct(struct_list: List[List[str]]) -> List[str]:
    mats = set()
    for seq in struct_list:
        for s in seq:
            if "_" in s:
                mats.add(s.split("_", 1)[0])
    return sorted(list(mats))


# -------------------------
# spectrum slicing
# -------------------------
def slice_spec(spec: np.ndarray, spec_type: str) -> np.ndarray:
    """
    spec stored as [R...,T...]
    spec_type:
      - R: first half
      - T: second half
      - R_T: all
    """
    spec_type = spec_type.upper()
    if spec_type == "R_T":
        return spec
    if spec.ndim != 1:
        raise ValueError("slice_spec expects 1D spec vector.")
    L = spec.shape[0]
    if L % 2 != 0:
        raise ValueError(f"spec length {L} not even, cannot split into R/T.")
    W = L // 2
    if spec_type == "R":
        return spec[:W]
    if spec_type == "T":
        return spec[W:]
    raise ValueError(f"Unknown spec_type: {spec_type}")


# -------------------------
# nk loader
# -------------------------
def load_nk_torch(nk_dir: str, materials: List[str], wavelengths_um: np.ndarray) -> Dict[str, torch.Tensor]:
    nk = {}
    for mat in materials:
        path = os.path.join(nk_dir, f"{mat}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing nk csv: {path}")

        df = pd.read_csv(path).dropna()
        wl = df["wl"].values.astype(np.float64)
        n = df["n"].values.astype(np.float64)
        k = df["k"].values.astype(np.float64)

        n_interp = interp1d(wl, n, bounds_error=False, fill_value="extrapolate")
        k_interp = interp1d(wl, k, bounds_error=False, fill_value="extrapolate")

        nk_np = n_interp(wavelengths_um) + 1j * k_interp(wavelengths_um)
        nk[mat] = torch.tensor(nk_np, dtype=COMPLEX_DTYPE, device=DEVICE)
    return nk


# -------------------------
# boundary condition helpers
# -------------------------
def choose_incident_n(W: int) -> torch.Tensor:
    if INC_MEDIUM == "air":
        return torch.ones((W,), dtype=COMPLEX_DTYPE, device=DEVICE)
    raise ValueError(f"Unsupported INC_MEDIUM={INC_MEDIUM}")

def choose_exit_n(nk_dict: Dict[str, torch.Tensor], W: int) -> torch.Tensor:
    if EXIT_MEDIUM == "air":
        return torch.ones((W,), dtype=COMPLEX_DTYPE, device=DEVICE)

    if EXIT_MEDIUM == "substrate":
        if SUBSTRATE not in nk_dict:
            raise KeyError(f"SUBSTRATE={SUBSTRATE} nk not loaded. Please ensure nk_dir has {SUBSTRATE}.csv")
        nk_out = nk_dict[SUBSTRATE]
        if FORCE_SUBSTRATE_K0:
            nk_out = torch.real(nk_out).to(REAL_DTYPE).to(COMPLEX_DTYPE) + 0.0j
        return nk_out

    raise ValueError(f"Unsupported EXIT_MEDIUM={EXIT_MEDIUM}")


# -------------------------
# tmm_fast packing
# -------------------------
def _pack_batch_to_tmm_fast(
    batch_mats: List[List[str]],
    batch_thks_nm: List[List[float]],
    nk_dict_torch: Dict[str, torch.Tensor],
    wl_m: torch.Tensor,  # [W]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    n: [B, Lmax+2, W], d: [B, Lmax+2]
    incidence: INC_MEDIUM
    exit: EXIT_MEDIUM (substrate -> SUBSTRATE; optionally k=0)
    """
    B = len(batch_mats)
    W = wl_m.shape[0]
    Lmax = max(len(x) for x in batch_mats) if B > 0 else 0

    d = torch.zeros((B, Lmax + 2), dtype=REAL_DTYPE, device=DEVICE)
    d[:, 0] = float("inf")
    d[:, -1] = float("inf")

    n = torch.empty((B, Lmax + 2, W), dtype=COMPLEX_DTYPE, device=DEVICE)

    n_in = choose_incident_n(W)
    n_out = choose_exit_n(nk_dict_torch, W)

    n[:, 0, :] = n_in.unsqueeze(0).expand(B, W)
    n[:, -1, :] = n_out.unsqueeze(0).expand(B, W)

    for bi in range(B):
        mats = batch_mats[bi]
        thks = batch_thks_nm[bi]
        L = len(mats)

        if L > 0:
            d[bi, 1:1 + L] = torch.tensor(thks, dtype=REAL_DTYPE, device=DEVICE) * 1e-9
            for li, m in enumerate(mats, start=1):
                if m not in nk_dict_torch:
                    raise KeyError(
                        f"Material '{m}' not found in nk_dict_torch. "
                        f"Either missing nk csv or you didn't load it."
                    )
                n[bi, li, :] = nk_dict_torch[m]

        # pad: thickness=0, n copy last real layer (safe)
        if L < Lmax and L > 0:
            n_pad = n[bi, L, :].clone()
            n[bi, 1 + L:1 + Lmax, :] = n_pad

        # if empty structure (shouldn't happen), fill with n_in
        if L == 0 and Lmax > 0:
            n[bi, 1:1 + Lmax, :] = n_in.unsqueeze(0).expand(Lmax, W)

    return n, d


@torch.no_grad()
def calc_spec_tmmfast_batch(
    batch_mats: List[List[str]],
    batch_thks_nm: List[List[float]],
    nk_dict_torch: Dict[str, torch.Tensor],
    wl_m: torch.Tensor,
    theta_rad: torch.Tensor,
    pol: str = "s",
    spec_type: str = "R_T",
) -> np.ndarray:
    """
    return spec: [B, D] where D depends on spec_type (R/T/R_T)
    Internally always compute R/T then select.
    """
    n, d = _pack_batch_to_tmm_fast(batch_mats, batch_thks_nm, nk_dict_torch, wl_m)
    out = tmm_fast.coh_tmm(pol, n, d, theta_rad, wl_m)

    R = out["R"]
    T = out["T"]
    if R.ndim == 3:
        R = R[:, 0, :]
        T = T[:, 0, :]
    elif R.ndim != 2:
        raise RuntimeError(f"Unexpected R shape: {tuple(R.shape)}")

    st = spec_type.upper()
    if st == "R_T":
        spec = torch.cat([R, T], dim=-1)
    elif st == "R":
        spec = R
    elif st == "T":
        spec = T
    else:
        raise ValueError(f"Unknown spec_type: {spec_type}")

    return spec.detach().cpu().float().numpy()


# -------------------------
# decode
# -------------------------
@torch.no_grad()
def greedy_decode(model, struc_index_dict, struc_word_dict, spec_target, max_len, start_symbol="BOS"):
    bos_id = struc_word_dict[start_symbol]
    ys = torch.ones(1, 1, dtype=torch.long, device=DEVICE).fill_(bos_id)

    spec_np = np.asarray(spec_target, dtype=np.float32)
    src = torch.from_numpy(spec_np).to(DEVICE)[None, None, :]
    src_mask = None

    out_tokens = []
    for _ in range(max_len - 1):
        trg_mask = Variable(subsequent_mask(ys.size(1)).type_as(src.data)).to(DEVICE)
        out = model(src, Variable(ys), src_mask, trg_mask)
        prob = model.generator(out[:, -1])

        next_id = int(prob.argmax(dim=1).item())
        ys = torch.cat([ys, torch.ones(1, 1, dtype=torch.long, device=DEVICE).fill_(next_id)], dim=1)

        sym = struc_index_dict.get(next_id, "UNK")
        if sym == "EOS":
            break
        out_tokens.append(sym)

    return out_tokens


# -------------------------
# bucket helpers
# -------------------------
def layers_bucket(L: int) -> str:
    if L <= 1: return "L=1"
    if L == 2: return "L=2"
    if L == 3: return "L=3"
    if 4 <= L <= 5: return "L=4-5"
    if 6 <= L <= 10: return "L=6-10"
    if 11 <= L <= 18: return "L=11-18"
    return "L>=19"


# -------------------------
# curve plotting (server-safe)
# -------------------------
def _safe_makedirs(p: str):
    os.makedirs(p, exist_ok=True)

def _sanitize_name(s: str) -> str:
    s = str(s)
    s = s.replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")
    s = s.replace("|", "_").replace(",", "_").replace(";", "_")
    return s[:120]

def _curve_metrics(y_hat: np.ndarray, y_gt: np.ndarray, x: np.ndarray) -> Dict[str, float]:
    """Scalar metrics on 1D curve y(x)."""
    y_hat = np.asarray(y_hat, dtype=np.float64)
    y_gt  = np.asarray(y_gt, dtype=np.float64)
    x     = np.asarray(x, dtype=np.float64)

    d = y_hat - y_gt
    mae = float(np.mean(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d * d)))
    mx = float(np.max(np.abs(d)))

    # integrate absolute error over wavelength
    area = float(np.trapz(np.abs(d), x))

    # correlation
    if np.std(y_hat) < 1e-12 or np.std(y_gt) < 1e-12:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(y_hat, y_gt)[0, 1])

    # PSNR with dynamic range from GT
    y_min = float(np.min(y_gt))
    y_max = float(np.max(y_gt))
    dr = max(y_max - y_min, 1e-12)
    psnr = float(20.0 * np.log10(dr) - 10.0 * np.log10(max(rmse * rmse, 1e-24)))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAX": mx,
        "AreaAbs": area,
        "Corr": corr,
        "PSNR": psnr,
    }

def _plot_curve_and_err(
    wavelengths_um: np.ndarray,
    gt_mean: np.ndarray,
    oracle_mean: np.ndarray,
    pred_mean: np.ndarray,
    out_dir: str,
    group_name: str,
    title_prefix: str,
    ylabel: str,
    save_csv: bool = False,
    save_signed_err: bool = True,
    save_metrics_json: bool = True,
):
    _safe_makedirs(out_dir)

    x = np.asarray(wavelengths_um, dtype=np.float64)
    gt = np.asarray(gt_mean, dtype=np.float64)
    oc = np.asarray(oracle_mean, dtype=np.float64)
    pd = np.asarray(pred_mean, dtype=np.float64)

    # ---- compute metrics (scalar) ----
    m_oracle = _curve_metrics(oc, gt, x)
    m_pred   = _curve_metrics(pd, gt, x)

    def _fmt(m: Dict[str, float]) -> str:
        return (
            f"MAE={m['MAE']:.4g}\n"
            f"RMSE={m['RMSE']:.4g}\n"
            f"MAX={m['MAX']:.4g}\n"
            f"Area|Δ|={m['AreaAbs']:.4g}\n"
            f"Corr={m['Corr']:.4g}\n"
            f"PSNR={m['PSNR']:.3g}"
        )

    txt = "Oracle vs GT\n" + _fmt(m_oracle) + "\n\nPred vs GT\n" + _fmt(m_pred)

    # =========================
    # mean curves
    # =========================
    plt.figure()
    plt.plot(x, gt, label="GT(dev_spec)")
    plt.plot(x, oc, label="Oracle(TMM(gt))")
    plt.plot(x, pd, label="Pred(TMM(pred))")
    plt.xlabel("Wavelength (um)")
    plt.ylabel(ylabel)
    plt.title(f"{title_prefix} - {group_name}")
    plt.legend()

    plt.gca().text(
        0.99, 0.99, txt,
        transform=plt.gca().transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.85)
    )

    plt.tight_layout()
    f1 = os.path.join(out_dir, f"mean_curve__{group_name}.png")
    plt.savefig(f1, dpi=200)
    plt.close()

    # =========================
    # abs error curves
    # =========================
    abs_oc = np.abs(oc - gt)
    abs_pd = np.abs(pd - gt)

    plt.figure()
    plt.plot(x, abs_oc, label="|Oracle-GT|")
    plt.plot(x, abs_pd, label="|Pred-GT|")
    plt.xlabel("Wavelength (um)")
    plt.ylabel(f"|Δ| ({ylabel})")
    plt.title(f"Mean Abs Error - {group_name}")
    plt.legend()

    txt2 = (
        f"Oracle: MAE={m_oracle['MAE']:.4g}, RMSE={m_oracle['RMSE']:.4g}, MAX={m_oracle['MAX']:.4g}, Area={m_oracle['AreaAbs']:.4g}\n"
        f"Pred  : MAE={m_pred['MAE']:.4g}, RMSE={m_pred['RMSE']:.4g}, MAX={m_pred['MAX']:.4g}, Area={m_pred['AreaAbs']:.4g}"
    )
    plt.gca().text(
        0.01, 0.99, txt2,
        transform=plt.gca().transAxes,
        ha="left", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.85)
    )

    plt.tight_layout()
    f2 = os.path.join(out_dir, f"mean_abs_err__{group_name}.png")
    plt.savefig(f2, dpi=200)
    plt.close()

    # =========================
    # signed error curves (optional)
    # =========================
    if save_signed_err:
        plt.figure()
        plt.plot(x, (oc - gt), label="Oracle-GT")
        plt.plot(x, (pd - gt), label="Pred-GT")
        plt.axhline(0.0, linewidth=1)
        plt.xlabel("Wavelength (um)")
        plt.ylabel(f"Δ ({ylabel})")
        plt.title(f"Mean Signed Error - {group_name}")
        plt.legend()
        plt.tight_layout()
        f3 = os.path.join(out_dir, f"mean_signed_err__{group_name}.png")
        plt.savefig(f3, dpi=200)
        plt.close()

    # =========================
    # CSV + metrics json (optional)
    # =========================
    if save_csv:
        csv_path = os.path.join(out_dir, f"mean_curve__{group_name}.csv")
        df = pd.DataFrame({
            "wavelength_um": x,
            "gt_mean": gt,
            "oracle_mean": oc,
            "pred_mean": pd,
            "abs_err_oracle": np.abs(oc - gt),
            "abs_err_pred": np.abs(pd - gt),
            "signed_err_oracle": (oc - gt),
            "signed_err_pred": (pd - gt),
        })
        df.to_csv(csv_path, index=False)

    if save_metrics_json:
        metrics_path = os.path.join(out_dir, f"metrics__{group_name}.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"oracle_vs_gt": m_oracle, "pred_vs_gt": m_pred}, f, ensure_ascii=False, indent=2)

def _plot_groups_mean_curves(
    wavelengths_um: np.ndarray,
    gt_spec_arr: np.ndarray,      # [N, D]
    oracle_spec_arr: np.ndarray,  # [N, D]
    pred_spec_arr: np.ndarray,    # [N, D]
    group_keys: List[str],
    out_dir: str,
    group_prefix: str,
    plot_min_count: int,
    plot_max_groups: int,
    save_csv: bool,
    spec_type: str,
    save_signed_err: bool,
    save_metrics_json: bool,
):
    """
    自动处理 spec_type:
      - R / T: D=W
      - R_T: D=2W => 分别保存 R 与 T 两套图
    """
    _safe_makedirs(out_dir)
    mp = defaultdict(list)
    for i, g in enumerate(group_keys):
        mp[str(g)].append(i)

    groups_sorted = sorted(mp.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    plotted = 0

    for g, ids in groups_sorted:
        if plotted >= plot_max_groups:
            break
        if len(ids) < plot_min_count:
            continue

        ids_np = np.asarray(ids, dtype=np.int64)

        gt_mean = gt_spec_arr[ids_np].mean(axis=0)
        oracle_mean = oracle_spec_arr[ids_np].mean(axis=0)
        pred_mean = pred_spec_arr[ids_np].mean(axis=0)

        group_name = _sanitize_name(f"{group_prefix}_{g}__n{len(ids)}")

        st = spec_type.upper()
        if st in ("R", "T"):
            if gt_mean.shape[0] != wavelengths_um.shape[0]:
                raise ValueError(
                    f"[Plot] spec_dim={gt_mean.shape[0]} but W={wavelengths_um.shape[0]} for spec_type={st}."
                )
            _plot_curve_and_err(
                wavelengths_um=wavelengths_um,
                gt_mean=gt_mean,
                oracle_mean=oracle_mean,
                pred_mean=pred_mean,
                out_dir=out_dir,
                group_name=group_name,
                title_prefix=f"Mean Curve ({st})",
                ylabel=st,
                save_csv=save_csv,
                save_signed_err=save_signed_err,
                save_metrics_json=save_metrics_json,
            )
        elif st == "R_T":
            W = wavelengths_um.shape[0]
            if gt_mean.shape[0] != 2 * W:
                raise ValueError(f"[Plot] spec_dim={gt_mean.shape[0]} != 2W={2*W} for spec_type=R_T.")
            # R
            _plot_curve_and_err(
                wavelengths_um=wavelengths_um,
                gt_mean=gt_mean[:W],
                oracle_mean=oracle_mean[:W],
                pred_mean=pred_mean[:W],
                out_dir=out_dir,
                group_name=group_name + "__R",
                title_prefix="Mean Curve (R)",
                ylabel="R",
                save_csv=save_csv,
                save_signed_err=save_signed_err,
                save_metrics_json=save_metrics_json,
            )
            # T
            _plot_curve_and_err(
                wavelengths_um=wavelengths_um,
                gt_mean=gt_mean[W:],
                oracle_mean=oracle_mean[W:],
                pred_mean=pred_mean[W:],
                out_dir=out_dir,
                group_name=group_name + "__T",
                title_prefix="Mean Curve (T)",
                ylabel="T",
                save_csv=save_csv,
                save_signed_err=save_signed_err,
                save_metrics_json=save_metrics_json,
            )
        else:
            raise ValueError(f"Unknown spec_type: {spec_type}")

        plotted += 1

    print(f"[CurvePlot] Saved {plotted} groups to: {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--dev_struct", type=str, required=True)
    ap.add_argument("--dev_spec", type=str, required=True)
    ap.add_argument("--dev_meta", type=str, default=None, help="Optional meta_dev.pkl (for type/layers report).")

    ap.add_argument("--nk_dir", type=str, default="./dataset/data")
    ap.add_argument("--lambda0", type=float, default=0.9)
    ap.add_argument("--lambda1", type=float, default=1.7)
    ap.add_argument("--step_um", type=float, default=0.005)

    ap.add_argument("--spec_type", type=str, default="R_T", help="R / T / R_T")
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--num_eval", type=int, default=2000, help="<=0 means all")
    ap.add_argument("--print_k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pair_topk", type=int, default=10)
    ap.add_argument("--token_topk", type=int, default=10)
    ap.add_argument("--tmm_batch", type=int, default=256)

    # ===== curve plot options =====
    ap.add_argument("--out_dir", type=str, default="./eval_curves", help="Where to save curve pngs/csvs/json.")
    ap.add_argument("--plot_mean_by_type", action="store_true", help="Save mean curves per type.")
    ap.add_argument("--plot_mean_by_layers", action="store_true", help="Save mean curves per layer-bucket.")
    ap.add_argument("--plot_max_groups", type=int, default=50, help="Max number of groups to plot (avoid too many files).")
    ap.add_argument("--plot_min_count", type=int, default=20, help="Only plot groups with >= this many samples.")
    ap.add_argument("--save_curve_csv", action="store_true", help="Also save mean curves to csv.")
    ap.add_argument("--save_signed_err", action="store_true", help="Also save mean_signed_err png.")
    ap.add_argument("--save_metrics_json", action="store_true", help="Also save per-plot metrics json.")

    args = ap.parse_args()
    set_seed(args.seed)

    # load ckpt
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["configs"]

    model = make_model_I(
        cfg.spec_dim, cfg.struc_dim, cfg.layers, cfg.d_model, cfg.d_ff, cfg.head_num, cfg.dropout
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    struc_word_dict = cfg.struc_word_dict
    struc_index_dict = cfg.struc_index_dict

    dev_struct = load_pickle(args.dev_struct)
    dev_spec_all = np.asarray(load_pickle(args.dev_spec), dtype=np.float32)

    N = len(dev_spec_all)
    if N != len(dev_struct):
        raise ValueError(f"len(dev_spec)={N} != len(dev_struct)={len(dev_struct)}")

    # load meta if available
    dev_meta = None
    if args.dev_meta is not None and os.path.exists(args.dev_meta):
        dev_meta = load_pickle(args.dev_meta)
        if len(dev_meta) != N:
            print(f"[WARN] meta size mismatch: len(meta)={len(dev_meta)} vs N={N}. Will ignore meta.")
            dev_meta = None

    # wavelength grid
    n_pts = int(round((args.lambda1 - args.lambda0) / args.step_um)) + 1
    wavelengths_um = np.linspace(args.lambda0, args.lambda1, n_pts)

    # slice spec according to spec_type
    dev_spec = np.stack([slice_spec(dev_spec_all[i], args.spec_type) for i in range(N)], axis=0)
    spec_dim = dev_spec.shape[1]
    if spec_dim != cfg.spec_dim:
        raise ValueError(
            f"spec_dim mismatch after slicing: dev_spec={spec_dim}, cfg={cfg.spec_dim} "
            f"(did you train with spec_type={args.spec_type}?)"
        )

    # indices to eval
    idxs = list(range(N))
    total_dev = N
    if args.num_eval > 0 and args.num_eval < N:
        idxs = random.sample(idxs, args.num_eval)

    # collect materials from GT
    mats_in_data = set(collect_materials_from_struct([dev_struct[i] for i in idxs]))

    # must include substrate if using substrate exit
    if EXIT_MEDIUM == "substrate":
        mats_in_data.add(SUBSTRATE)

    # pre-load nk for GT mats; pred mats loaded on-the-fly
    materials_to_load = sorted(list(mats_in_data))
    nk_dict_torch = load_nk_torch(args.nk_dir, materials_to_load, wavelengths_um)

    wl_m = torch.tensor(wavelengths_um * 1e-6, dtype=REAL_DTYPE, device=DEVICE)
    theta_rad = torch.tensor([0.0], dtype=REAL_DTYPE, device=DEVICE)

    # containers
    oracle_mats, oracle_thks, oracle_targets = [], [], []
    pred_mats, pred_thks, pred_targets = [], [], []
    pred_tokens_list = []
    pair_counter = Counter()
    token_counter = Counter()

    # bookkeeping for reports
    type_list = []
    layers_list = []

    for j, ii in enumerate(idxs):
        spec_target = dev_spec[ii]
        gt_tokens = dev_struct[ii]

        mats_gt, thks_gt = parse_structure_tokens(gt_tokens)
        oracle_mats.append(mats_gt)
        oracle_thks.append(thks_gt)
        oracle_targets.append(spec_target)

        # meta
        if dev_meta is not None:
            mt = dev_meta[ii]
            t = mt.get("type", mt.get("family", "UNKNOWN"))
            nl = int(mt.get("num_layers", len(gt_tokens)))
        else:
            t = "UNKNOWN"
            nl = len(gt_tokens)
        type_list.append(t)
        layers_list.append(nl)

        # pred
        pred_tokens = greedy_decode(model, struc_index_dict, struc_word_dict, spec_target, args.max_len)
        pred_tokens_list.append(pred_tokens)

        pair_counter[infer_pair_name(pred_tokens)] += 1
        for tok in pred_tokens:
            if "_" in tok:
                token_counter[tok] += 1

        mats_pred, thks_pred = parse_structure_tokens(pred_tokens)
        pred_mats.append(mats_pred)
        pred_thks.append(thks_pred)
        pred_targets.append(spec_target)

    # helper to ensure nk coverage if pred introduces new mats
    def ensure_nk_for_pred(batch_pred_mats: List[List[str]]):
        nonlocal nk_dict_torch, materials_to_load
        new_mats = set()
        for mats in batch_pred_mats:
            for m in mats:
                if m not in nk_dict_torch:
                    new_mats.add(m)
        if EXIT_MEDIUM == "substrate" and SUBSTRATE not in nk_dict_torch:
            new_mats.add(SUBSTRATE)

        if new_mats:
            new_list = sorted(list(new_mats))
            nk_new = load_nk_torch(args.nk_dir, new_list, wavelengths_um)
            nk_dict_torch.update(nk_new)
            materials_to_load = sorted(list(set(materials_to_load).union(new_mats)))

    # batch MAE
    oracle_mae_list = []
    pred_mae_list = []

    # ===== store spectra for curve comparison =====
    oracle_spec_list = []  # each is [D]
    pred_spec_list = []    # each is [D]
    gt_spec_list = []      # each is [D] (dev_spec sliced)

    B = max(1, int(args.tmm_batch))

    for st in range(0, len(idxs), B):
        ed = min(len(idxs), st + B)

        # oracle
        spec_gt = calc_spec_tmmfast_batch(
            oracle_mats[st:ed], oracle_thks[st:ed],
            nk_dict_torch, wl_m, theta_rad,
            pol=POL, spec_type=args.spec_type
        )
        targets = np.asarray(oracle_targets[st:ed], dtype=np.float32)
        mae = np.mean(np.abs(spec_gt - targets), axis=1)
        oracle_mae_list.extend(mae.tolist())

        oracle_spec_list.extend(spec_gt.tolist())
        gt_spec_list.extend(targets.tolist())

        # pred
        ensure_nk_for_pred(pred_mats[st:ed])
        spec_pd = calc_spec_tmmfast_batch(
            pred_mats[st:ed], pred_thks[st:ed],
            nk_dict_torch, wl_m, theta_rad,
            pol=POL, spec_type=args.spec_type
        )
        targets2 = np.asarray(pred_targets[st:ed], dtype=np.float32)
        mae2 = np.mean(np.abs(spec_pd - targets2), axis=1)
        pred_mae_list.extend(mae2.tolist())

        pred_spec_list.extend(spec_pd.tolist())

    oracle_mae = np.asarray(oracle_mae_list, dtype=np.float32)
    pred_mae = np.asarray(pred_mae_list, dtype=np.float32)

    # -------------------------
    # print samples
    # -------------------------
    for j in range(min(args.print_k, len(idxs))):
        ii = idxs[j]
        gt_tokens = dev_struct[ii]
        pred_tokens = pred_tokens_list[j]
        t = type_list[j]
        nl = layers_list[j]

        print(f"\n---- sample {ii} ----")
        print(f"type={t} | num_layers={nl}")
        print(f"GT   pair: {infer_pair_name(gt_tokens)} | len={len(gt_tokens)}")
        print(f"PRED pair: {infer_pair_name(pred_tokens)} | len={len(pred_tokens)}")
        print("GT   head:", gt_tokens[:10], "..." if len(gt_tokens) > 10 else "")
        print("PRED head:", pred_tokens[:10], "..." if len(pred_tokens) > 10 else "")
        print(f"Oracle MAE: {oracle_mae_list[j]:.6f}")
        print(f"Pred   MAE: {pred_mae_list[j]:.6f}")

    # -------------------------
    # overall summary
    # -------------------------
    print("\n==================== OVERALL ====================")
    print(f"eval samples: {len(idxs)} / total_dev={total_dev}")
    print(f"spec_type={args.spec_type} | pol={POL}")
    print(f"INC_MEDIUM={INC_MEDIUM} | EXIT_MEDIUM={EXIT_MEDIUM} | SUBSTRATE={SUBSTRATE} | FORCE_SUBSTRATE_K0={FORCE_SUBSTRATE_K0}")
    print(f"grid: lambda0={args.lambda0} lambda1={args.lambda1} step={args.step_um} | W={len(wavelengths_um)}")
    print(f"loaded_materials={len(materials_to_load)} | nk_dir={args.nk_dir}")

    print("\n[Oracle MAE] (GT -> TMM_FAST vs dev_spec)")
    print(f"  mean  : {oracle_mae.mean():.6f}")
    print(f"  median: {np.median(oracle_mae):.6f}")
    print(f"  p90   : {np.quantile(oracle_mae, 0.90):.6f}")

    print("\n[Pred MAE] (Pred -> TMM_FAST vs dev_spec)")
    print(f"  mean  : {pred_mae.mean():.6f}")
    print(f"  median: {np.median(pred_mae):.6f}")
    print(f"  p90   : {np.quantile(pred_mae, 0.90):.6f}")

    # -------------------------
    # by type / by layers bucket
    # -------------------------
    by_type = defaultdict(list)
    by_bucket = defaultdict(list)
    for m, t, nl in zip(pred_mae_list, type_list, layers_list):
        by_type[str(t)].append(float(m))
        by_bucket[layers_bucket(int(nl))].append(float(m))

    if any(k != "UNKNOWN" for k in by_type.keys()):
        print("\n================= BY TYPE =================")
        for k in sorted(by_type.keys()):
            arr = np.asarray(by_type[k], dtype=np.float32)
            print(f"{k:8s}  n={len(arr):6d}  mean={arr.mean():.6f}  median={np.median(arr):.6f}  p90={np.quantile(arr,0.90):.6f}")

    print("\n============= BY NUM_LAYERS BUCKET =============")
    bucket_order = ["L=1","L=2","L=3","L=4-5","L=6-10","L=11-18","L>=19"]
    for b in bucket_order:
        if b not in by_bucket: continue
        arr = np.asarray(by_bucket[b], dtype=np.float32)
        print(f"{b:6s}  n={len(arr):6d}  mean={arr.mean():.6f}  median={np.median(arr):.6f}  p90={np.quantile(arr,0.90):.6f}")

    # -------------------------
    # curve comparison (save only, no show)
    # -------------------------
    if args.plot_mean_by_type or args.plot_mean_by_layers:
        _safe_makedirs(args.out_dir)

        gt_spec_arr = np.asarray(gt_spec_list, dtype=np.float32)         # [N_eval, D]
        oracle_spec_arr = np.asarray(oracle_spec_list, dtype=np.float32) # [N_eval, D]
        pred_spec_arr = np.asarray(pred_spec_list, dtype=np.float32)     # [N_eval, D]

        if args.plot_mean_by_type:
            out_dir_type = os.path.join(args.out_dir, "by_type")
            _plot_groups_mean_curves(
                wavelengths_um=wavelengths_um,
                gt_spec_arr=gt_spec_arr,
                oracle_spec_arr=oracle_spec_arr,
                pred_spec_arr=pred_spec_arr,
                group_keys=[str(x) for x in type_list],
                out_dir=out_dir_type,
                group_prefix="type",
                plot_min_count=args.plot_min_count,
                plot_max_groups=args.plot_max_groups,
                save_csv=args.save_curve_csv,
                spec_type=args.spec_type,
                save_signed_err=args.save_signed_err,
                save_metrics_json=args.save_metrics_json,
            )

        if args.plot_mean_by_layers:
            out_dir_layers = os.path.join(args.out_dir, "by_layers_bucket")
            layer_bucket_list = [layers_bucket(int(x)) for x in layers_list]
            _plot_groups_mean_curves(
                wavelengths_um=wavelengths_um,
                gt_spec_arr=gt_spec_arr,
                oracle_spec_arr=oracle_spec_arr,
                pred_spec_arr=pred_spec_arr,
                group_keys=layer_bucket_list,
                out_dir=out_dir_layers,
                group_prefix="layers",
                plot_min_count=args.plot_min_count,
                plot_max_groups=args.plot_max_groups,
                save_csv=args.save_curve_csv,
                spec_type=args.spec_type,
                save_signed_err=args.save_signed_err,
                save_metrics_json=args.save_metrics_json,
            )

    # -------------------------
    # pred distributions
    # -------------------------
    print("\n============= PRED DISTRIBUTIONS =============")
    print("[Pred pair top %d]" % args.pair_topk)
    total = sum(pair_counter.values())
    for k, v in pair_counter.most_common(args.pair_topk):
        print(f"  {k:20s} {v:6d} ({v/max(total,1):.3%})")

    print("\n[Pred token top %d]" % args.token_topk)
    total2 = sum(token_counter.values())
    for k, v in token_counter.most_common(args.token_topk):
        print(f"  {k:20s} {v:6d} ({v/max(total2,1):.3%})")

    print("==================================================")


if __name__ == "__main__":
    main()


"""
示例：

python eval_all.py \
  --ckpt saved_models/optogpt/all_new/best.pt \
  --dev_struct ./dataset/all_new/Structure_dev.pkl \
  --dev_spec   ./dataset/all_new/Spectrum_dev.pkl \
  --dev_meta   ./dataset/all_new/meta_dev.pkl \
  --nk_dir     ./dataset/data \
  --spec_type  R_T \
  --max_len 22 \
  --num_eval 2000 \
  --print_k 8 \
  --tmm_batch 256 \
  --plot_mean_by_type \
  --plot_mean_by_layers \
  --out_dir ./eval_curves_all_new_with_pic \
  --plot_min_count 20 \
  --plot_max_groups 50 \
  --save_curve_csv \
  --save_signed_err \
  --save_metrics_json
"""
