# eval_dbr_mae.py
# DBR eval (MAE) aligned with your dataset generation:
# - WAVELENGTHS: np.linspace(0.9, 1.7, int(round((1.7-0.9)/0.005))+1)
# - wl_nm: np.round(WAVELENGTHS*1000).astype(int)
# - forward: coh_tmm, air / layers / air
# - structure token: "Mat_123" saved with int(round(thickness_nm))
# - spectrum target: [R..., T...] (len = 2*len(WAVELENGTHS)) saved in Spectrum_dev.pkl

import os
import re
import argparse
import pickle as pkl
from typing import List, Dict, Tuple, Any

import numpy as np
import torch

import pandas as pd
from scipy.interpolate import interp1d
from tmm import coh_tmm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== your project import =====
from core.models.transformer import make_model_I


# =========================
# 0) wavelengths (must match your generator)
# =========================
WAVELENGTHS = np.linspace(0.9, 1.7, int(round((1.7 - 0.9) / 0.005)) + 1).astype(np.float64)  # um
WL_NM = np.round(WAVELENGTHS * 1000).astype(int)  # nm
SPEC_DIM_EXPECT = 2 * len(WAVELENGTHS)  # 322


# =========================
# 1) utils: io
# =========================
def load_pickle(path: str):
    with open(path, "rb") as f:
        return pkl.load(f)


def load_dev_data(dataset_dir: str) -> Tuple[List[List[str]], np.ndarray]:
    struct_path = os.path.join(dataset_dir, "Structure_dev.pkl")
    spec_path = os.path.join(dataset_dir, "Spectrum_dev.pkl")

    if not os.path.exists(struct_path):
        raise FileNotFoundError(f"Missing: {struct_path}")
    if not os.path.exists(spec_path):
        raise FileNotFoundError(f"Missing: {spec_path}")

    structures = load_pickle(struct_path)  # list[list[str]]
    spectra = np.array(load_pickle(spec_path), dtype=np.float32)  # (N, 2W)

    if spectra.ndim != 2:
        raise ValueError(f"Spectrum_dev should be 2D array-like, got shape {spectra.shape}")
    if spectra.shape[1] != SPEC_DIM_EXPECT:
        raise ValueError(
            f"Spectrum dim mismatch: got {spectra.shape[1]}, expect {SPEC_DIM_EXPECT} "
            f"(0.9–1.7 um, step=0.005, R+T)."
        )
    if len(structures) != spectra.shape[0]:
        raise ValueError(f"Size mismatch: len(structure)={len(structures)} vs spectra={spectra.shape[0]}")
    return structures, spectra


# =========================
# 2) nk loader (csv: wl(um), n, k)
# =========================
def load_nk(materials: List[str], nk_dir: str, wavelengths_um: np.ndarray) -> Dict[str, np.ndarray]:
    nk = {}
    for mat in materials:
        path = os.path.join(nk_dir, f"{mat}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing nk csv: {path}")

        df = pd.read_csv(path).dropna()
        wl = df["wl"].values.astype(np.float64)  # um
        n = df["n"].values.astype(np.float64)
        k = df["k"].values.astype(np.float64)

        n_itp = interp1d(wl, n, bounds_error=False, fill_value="extrapolate")
        k_itp = interp1d(wl, k, bounds_error=False, fill_value="extrapolate")
        nk[mat] = n_itp(wavelengths_um) + 1j * k_itp(wavelengths_um)
    return nk


# =========================
# 3) DBR forward: air / layers / air
# =========================
_LAYER_RE = re.compile(r"^([A-Za-z0-9]+)_([0-9]+)$")

def parse_layers(tokens: List[str]) -> Tuple[List[str], List[float]]:
    """
    Keep only tokens like 'TiO2_123' (thickness nm is int-ish).
    Return mats, thks_nm.
    """
    mats, thks = [], []
    for t in tokens:
        m = _LAYER_RE.match(t.strip())
        if not m:
            continue
        mats.append(m.group(1))
        thks.append(float(m.group(2)))
    return mats, thks


def spectrum_dbr(materials: List[str], thickness_nm: List[float], nk_dict: Dict[str, np.ndarray],
                 pol: str = "s", theta_deg: float = 0.0) -> np.ndarray:
    d_list = [np.inf] + list(map(float, thickness_nm)) + [np.inf]
    th0 = np.deg2rad(theta_deg)

    R, T = [], []
    for i, lam_nm in enumerate(WL_NM):
        n_list = [1.0] + [nk_dict[m][i] for m in materials] + [1.0]
        res = coh_tmm(pol=pol, n_list=n_list, d_list=d_list, th_0=th0, lam_vac=float(lam_nm))
        R.append(res["R"])
        T.append(res["T"])
    return np.asarray(R + T, dtype=np.float32)  # (2W,)


# =========================
# 4) decode (greedy)
#    - Prefer configs.struc_word_dict / struc_index_dict like original analysis
#    - Requires model.encode/decode/generator. If your model differs, tell me the forward signature and I’ll adapt.
# =========================
def subsequent_mask(size: int) -> torch.Tensor:
    attn_shape = (1, size, size)
    subsequent = torch.triu(torch.ones(attn_shape, dtype=torch.bool), diagonal=1)
    return ~subsequent


def _get_special_id(word_dict: Dict[str, int], candidates: List[str], name: str) -> int:
    for c in candidates:
        if c in word_dict:
            return int(word_dict[c])
    raise KeyError(f"Cannot find {name} token in struc_word_dict. Tried: {candidates}")


@torch.no_grad()
def greedy_decode_structure(model, configs, spec_vec: np.ndarray, max_len: int = 256) -> List[str]:
    """
    Input: spec_vec shape (2W,)
    Output: list[str] predicted tokens (without BOS/EOS)
    """
    word_dict: Dict[str, int] = configs.struc_word_dict
    index_dict: Dict[int, str] = configs.struc_index_dict

    bos_id = _get_special_id(word_dict, ["BOS", "<BOS>", "bos"], "BOS")
    eos_id = _get_special_id(word_dict, ["EOS", "<EOS>", "eos"], "EOS")

    # src: (1, 1, D) for a "sequence length=1" continuous spectrum token
    src = torch.from_numpy(spec_vec).float().view(1, 1, -1).to(DEVICE)
    src_mask = torch.ones((1, 1, 1), dtype=torch.bool, device=DEVICE)

    # must have encode/decode/generator
    memory = model.encode(src, src_mask)
    ys = torch.tensor([[bos_id]], dtype=torch.long, device=DEVICE)

    out_tokens: List[str] = []
    for _ in range(max_len):
        tgt_mask = subsequent_mask(ys.size(1)).to(DEVICE)
        out = model.decode(memory, src_mask, ys, tgt_mask)          # (1, L, d_model)
        prob = model.generator(out[:, -1, :])                       # (1, vocab)
        next_id = int(torch.argmax(prob, dim=-1).item())

        if next_id == eos_id:
            break

        ys = torch.cat([ys, torch.tensor([[next_id]], device=DEVICE)], dim=1)
        tok = index_dict.get(next_id, "UNK")
        out_tokens.append(tok)

    return out_tokens


# =========================
# 5) eval MAE
# =========================
@torch.no_grad()
def eval_mae(model, configs,
             spectra_gt: np.ndarray,
             nk_dict: Dict[str, np.ndarray],
             max_samples: int = 200,
             max_len: int = 256,
             spec_type: str = "R_T") -> Dict[str, Any]:
    """
    spec_type:
      - R_T: MAE on [R..., T...]
      - R  : MAE only on R
      - T  : MAE only on T
    """
    N = min(max_samples, spectra_gt.shape[0])
    maes: List[float] = []
    bad_forward = 0
    empty_pred = 0

    for i in range(N):
        target = spectra_gt[i]  # (2W,)

        pred_tokens = greedy_decode_structure(model, configs, target, max_len=max_len)
        mats, thks = parse_layers(pred_tokens)
        if len(mats) == 0:
            empty_pred += 1
            # fallback: mae = 1 (or skip). Here we skip to avoid polluting results.
            continue

        try:
            pred_spec = spectrum_dbr(mats, thks, nk_dict)
        except Exception:
            bad_forward += 1
            continue

        if spec_type == "R":
            pred_use = pred_spec[:len(WAVELENGTHS)]
            gt_use = target[:len(WAVELENGTHS)]
        elif spec_type == "T":
            pred_use = pred_spec[len(WAVELENGTHS):]
            gt_use = target[len(WAVELENGTHS):]
        else:
            pred_use = pred_spec
            gt_use = target

        mae = float(np.mean(np.abs(pred_use - gt_use)))
        maes.append(mae)

    report = {
        "num_eval_requested": int(N),
        "num_eval_used": int(len(maes)),
        "mean_mae": float(np.mean(maes)) if maes else float("nan"),
        "mae_list": maes,
        "empty_pred": int(empty_pred),
        "bad_forward": int(bad_forward),
        "spec_type": spec_type,
        "spec_dim": int(spectra_gt.shape[1]),
        "num_wavelengths": int(len(WAVELENGTHS)),
        "wavelengths_um": WAVELENGTHS,   # helpful for later plotting
        "wavelengths_nm": WL_NM,
    }
    return report


# =========================
# 6) main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="path to your saved checkpoint .pt")
    ap.add_argument("--dataset_dir", type=str, default="./dataset", help="contains Structure_dev.pkl & Spectrum_dev.pkl")
    ap.add_argument("--nk_dir", type=str, default="./dataset/data", help="nk csv dir")
    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--spec_type", type=str, default="R_T", choices=["R_T", "R", "T"])
    ap.add_argument("--out", type=str, default="./results/eval_mae.pkl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # ---- load ckpt/config ----
    ckpt = torch.load(args.ckpt, map_location="cpu")
    configs = ckpt["configs"]

    # ---- build model ----
    model = make_model_I(
        configs.spec_dim,
        configs.struc_dim,
        configs.layers,
        configs.d_model,
        configs.d_ff,
        configs.head_num,
        configs.dropout
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    if int(configs.spec_dim) != SPEC_DIM_EXPECT:
        print(f"[WARN] configs.spec_dim={configs.spec_dim} != {SPEC_DIM_EXPECT}. "
              f"你的训练波长点可能和当前 eval 的 (0.9–1.7, 5nm, R+T) 不一致。")

    # ---- load dev ----
    structures_gt, spectra_gt = load_dev_data(args.dataset_dir)
    print(f"[INFO] Loaded dev: N={len(structures_gt)}, spectrum shape={spectra_gt.shape}")

    # ---- collect materials to load nk ----
    # 这里用 GT 结构里出现的材料集合；如果你预测可能超出 GT，可改成更大集合（比如 industry 白名单）
    mats_set = set()
    for st in structures_gt:
        for tok in st:
            m = tok.rsplit("_", 1)[0]
            mats_set.add(m)

    nk_dict = load_nk(sorted(mats_set), args.nk_dir, WAVELENGTHS)
    print(f"[INFO] Loaded nk for {len(nk_dict)} materials.")

    # ---- eval ----
    report = eval_mae(
        model=model,
        configs=configs,
        spectra_gt=spectra_gt,
        nk_dict=nk_dict,
        max_samples=args.max_samples,
        max_len=args.max_len,
        spec_type=args.spec_type
    )

    print("\n==== DBR Eval (MAE) ====")
    print("spec_type        :", report["spec_type"])
    print("num_eval_requested:", report["num_eval_requested"])
    print("num_eval_used    :", report["num_eval_used"])
    print("empty_pred       :", report["empty_pred"])
    print("bad_forward      :", report["bad_forward"])
    print("mean_mae         :", report["mean_mae"])

    with open(args.out, "wb") as f:
        pkl.dump(report, f)
    print("[Saved]", args.out)


if __name__ == "__main__":
    main()


# python eval_dbr_mae.py \
#   --ckpt saved_models/optogpt/in_paper/optogpt.pt \
#   --dataset_dir ./dataset \
#   --nk_dir ./dataset/data \
#   --max_samples 200 \
#   --spec_type R_T