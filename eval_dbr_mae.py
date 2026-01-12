import os
import time
import argparse
import pickle as pkl
import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable

from scipy.interpolate import interp1d
from tmm import coh_tmm

# ====== OptoGPT/你工程里的函数 ======
from core.models.transformer import make_model_I, subsequent_mask
from core.trains.train import count_params  # 你原版里用的
# 如果你 repo 里 count_params 不在这里，就删掉不影响 eval


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------
# 1) DBR: wavelengths
# --------------------------
def build_wavelengths(wmin=0.9, wmax=1.7, step=0.005):
    n = int(round((wmax - wmin) / step)) + 1
    return np.linspace(wmin, wmax, n).astype(np.float64)  # um


# --------------------------
# 2) load nk (wl in um)
# 每个材料一个 csv: wl,n,k
# --------------------------
def load_nk(materials, wavelengths_um, nk_dir):
    nk = {}
    for mat in materials:
        path = os.path.join(nk_dir, f"{mat}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"[nk missing] {path}")

        df = pd.read_csv(path).dropna()
        wl = df["wl"].values.astype(np.float64)  # um
        n = df["n"].values.astype(np.float64)
        k = df["k"].values.astype(np.float64)

        n_interp = interp1d(wl, n, bounds_error=False, fill_value="extrapolate")
        k_interp = interp1d(wl, k, bounds_error=False, fill_value="extrapolate")

        nk[mat] = (n_interp(wavelengths_um) + 1j * k_interp(wavelengths_um)).astype(np.complex128)
    return nk


# --------------------------
# 3) DBR spectrum (air - stack - air)
# 输入：materials(list[str]), thickness_nm(list[float])
# 输出：R+T, 形状 [2*Nw]
# --------------------------
def spectrum_dbr(materials, thickness_nm, wavelengths_um, nk_dict, pol="s", theta_deg=0.0):
    d_list = [np.inf] + list(thickness_nm) + [np.inf]
    th0 = np.deg2rad(theta_deg)

    wl_nm_list = np.round(wavelengths_um * 1000.0).astype(int)

    R = np.zeros((len(wl_nm_list),), dtype=np.float32)
    T = np.zeros((len(wl_nm_list),), dtype=np.float32)

    for i, wl_nm in enumerate(wl_nm_list):
        n_list = [1] + [nk_dict[m][i] for m in materials] + [1]
        res = coh_tmm(pol=pol, n_list=n_list, d_list=d_list, th_0=th0, lam_vac=wl_nm)
        R[i] = float(res["R"])
        T[i] = float(res["T"])

    return np.concatenate([R, T], axis=0).astype(np.float32)


# --------------------------
# 4) tokens -> (materials, thickness)
# token format: "TiO2_123"
# --------------------------
def return_mat_thick(struc_list):
    materials = []
    thickness = []
    for s in struc_list:
        if s in ("BOS", "EOS", "PAD"):
            continue
        if "_" not in s:
            continue
        m, t = s.split("_", 1)
        try:
            materials.append(m)
            thickness.append(float(t))  # nm
        except ValueError:
            continue
    return materials, thickness


# --------------------------
# 5) greedy_decode (按你原版逻辑写)
# 注意：不调用 model.encoder；直接 model(src, ys, src_mask, trg_mask)
# --------------------------
@torch.no_grad()
def greedy_decode(model, struc_word_dict, struc_index_dict, spec_target, max_len, start_symbol="BOS"):
    bos_id = struc_word_dict[start_symbol]
    ys = torch.ones(1, 1).fill_(bos_id).long().to(DEVICE)

    src = torch.tensor([spec_target]).unsqueeze(0).float().to(DEVICE)  # (1,1,spec_dim)
    src_mask = None

    struc_design = []
    for _ in range(max_len - 1):
        trg_mask = Variable(subsequent_mask(ys.size(1)).type_as(src.data))
        out = model(src, Variable(ys), src_mask, trg_mask.to(DEVICE))
        prob = model.generator(out[:, -1])  # (1, vocab)

        _, next_word = torch.max(prob, dim=1)
        next_id = int(next_word.item())

        ys = torch.cat([ys, torch.ones(1, 1, device=DEVICE, dtype=torch.long).fill_(next_id)], dim=1)

        sym = struc_index_dict[next_id]
        if sym == "EOS":
            break
        struc_design.append(sym)

    return struc_design


# --------------------------
# 6) 从 vocab 自动收集材料集合（用来加载 nk）
# --------------------------
def infer_materials_from_vocab(struc_word_dict):
    mats = set()
    for tok in struc_word_dict.keys():
        if tok in ("BOS", "EOS", "PAD"):
            continue
        if "_" in tok:
            mats.add(tok.split("_", 1)[0])
    return sorted(list(mats))


# --------------------------
# 7) eval main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="DBR trained checkpoint .pt")
    ap.add_argument("--nk_dir", type=str, required=True, help="folder of nk csv files (wl,n,k; wl in um)")
    ap.add_argument("--train_spec", type=str, default="", help="Spectrum_train.pkl (optional for best-in-data baseline)")
    ap.add_argument("--dev_spec", type=str, required=True, help="Spectrum_dev.pkl (or test spectrum)")
    ap.add_argument("--wmin", type=float, default=0.9)
    ap.add_argument("--wmax", type=float, default=1.7)
    ap.add_argument("--wstep", type=float, default=0.005)
    ap.add_argument("--max_eval", type=int, default=0, help="0=all, else first N")
    ap.add_argument("--out_csv", type=str, default="eval_dbr_mae.csv")
    args = ap.parse_args()

    wavelengths = build_wavelengths(args.wmin, args.wmax, args.wstep)
    nW = len(wavelengths)

    # ---- load ckpt + model ----
    a = torch.load(args.ckpt, map_location="cpu")
    cfg = a["configs"]

    torch.manual_seed(getattr(cfg, "seeds", 0))
    np.random.seed(getattr(cfg, "seeds", 0))

    model = make_model_I(
        cfg.spec_dim,
        cfg.struc_dim,
        cfg.layers,
        cfg.d_model,
        cfg.d_ff,
        cfg.head_num,
        cfg.dropout,
    ).to(DEVICE)
    model.load_state_dict(a["model_state_dict"])
    model.eval()

    try:
        print("Params:", count_params(model))
    except Exception:
        pass

    struc_word_dict = cfg.struc_word_dict
    struc_index_dict = cfg.struc_index_dict

    # ---- load spectra ----
    with open(args.dev_spec, "rb") as f:
        dev_spec = pkl.load(f)
    dev_spec = np.asarray(dev_spec, dtype=np.float32)

    if args.max_eval > 0:
        dev_spec = dev_spec[: args.max_eval]

    # shape check
    if dev_spec.shape[1] != 2 * nW:
        raise RuntimeError(
            f"[spec dim mismatch]\n"
            f"  dev_spec dim = {dev_spec.shape[1]}\n"
            f"  but wavelengths gives 2*nW = {2*nW}\n"
            f"  -> check (wmin,wmax,wstep) OR your dataset spectrum format."
        )

    # optional baseline pool (train_spec)
    train_spec = None
    if args.train_spec:
        with open(args.train_spec, "rb") as f:
            train_spec = np.asarray(pkl.load(f), dtype=np.float32)
        if train_spec.shape[1] != 2 * nW:
            print("[WARN] train_spec dim mismatch, baseline disabled.")
            train_spec = None

    # ---- load nk ----
    mats = infer_materials_from_vocab(struc_word_dict)
    nk_dict = load_nk(mats, wavelengths, args.nk_dir)

    # ---- eval loop ----
    maes_all, maes_R, maes_T = [], [], []
    base_maes = []  # best-in-data baseline (optional)
    rows = []

    t0 = time.time()
    for i in range(len(dev_spec)):
        target = dev_spec[i]  # [R...,T...]

        # 1) greedy generate structure
        pred_tokens = greedy_decode(
            model=model,
            struc_word_dict=struc_word_dict,
            struc_index_dict=struc_index_dict,
            spec_target=target,
            max_len=cfg.max_len,
            start_symbol="BOS",
        )

        # 2) forward physics: structure -> spectrum
        mats_i, thk_i = return_mat_thick(pred_tokens)
        if len(mats_i) == 0:
            pred_spec = np.zeros((2 * nW,), dtype=np.float32)
        else:
            pred_spec = spectrum_dbr(mats_i, thk_i, wavelengths, nk_dict)

        # 3) MAE
        mae_all = float(np.mean(np.abs(pred_spec - target)))
        mae_r = float(np.mean(np.abs(pred_spec[:nW] - target[:nW])))
        mae_t = float(np.mean(np.abs(pred_spec[nW:] - target[nW:])))

        maes_all.append(mae_all)
        maes_R.append(mae_r)
        maes_T.append(mae_t)

        # 4) baseline: best spectrum in train pool (optional)
        if train_spec is not None:
            # 原版的 temp_close: argmin(mean(abs(train_spec - spec_target)))
            base = float(np.min(np.mean(np.abs(train_spec - target[None, :]), axis=1)))
            base_maes.append(base)
        else:
            base = np.nan

        rows.append(
            dict(
                idx=i,
                mae_all=mae_all,
                mae_R=mae_r,
                mae_T=mae_t,
                baseline_best_in_train_mae=base,
                pred_layers=len(pred_tokens),
                pred_structure="|".join(pred_tokens),
            )
        )

        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(dev_spec)}] MAE_all={np.mean(maes_all):.6f}  time={time.time()-t0:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    print("\n===== DBR EVAL DONE =====")
    print(f"Samples: {len(dev_spec)}")
    print(f"MAE_all: {np.mean(maes_all):.6f}")
    print(f"MAE_R  : {np.mean(maes_R):.6f}")
    print(f"MAE_T  : {np.mean(maes_T):.6f}")
    if len(base_maes) > 0:
        print(f"Baseline(best-in-train) MAE_all: {np.mean(base_maes):.6f}")
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()


# python eval_dbr_mae.py \
#   --ckpt saved_models/optogpt/dbr/xxx_best.pt \
#   --nk_dir ./dataset/data \
#   --dev_spec ./dataset/Spectrum_dev.pkl \
#   --train_spec ./dataset/Spectrum_train.pkl \
#   --wmin 0.9 --wmax 1.7 --wstep 0.005 \
#   --out_csv eval_dbr_mae.csv