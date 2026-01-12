import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from scipy.interpolate import interp1d
from tmm import coh_tmm

# ====== 你原版 optogpt 的依赖（保持一致）======
from core.models.transformer import make_model_I, subsequent_mask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# 1) nk 加载（和你生成DBR数据一致）
# -------------------------
def load_nk(nk_dir, materials, wavelengths_um):
    nk = {}
    for mat in materials:
        path = os.path.join(nk_dir, f"{mat}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing nk csv: {path}")

        df = pd.read_csv(path).dropna()
        wl = df["wl"].values
        n = df["n"].values
        k = df["k"].values

        n_interp = interp1d(wl, n, bounds_error=False, fill_value="extrapolate")
        k_interp = interp1d(wl, k, bounds_error=False, fill_value="extrapolate")
        nk[mat] = n_interp(wavelengths_um) + 1j * k_interp(wavelengths_um)
    return nk


# -------------------------
# 2) DBR 光谱计算（与你生成代码 calc_RT 一致）
#    输出 [R..., T...] 维度 = 2*N
# -------------------------
def calc_RT(materials, thicknesses_nm, nk_dict, wavelengths_um, pol="s", theta_deg=0.0):
    R, T = [], []
    d_list = [np.inf] + list(thicknesses_nm) + [np.inf]
    th0 = np.deg2rad(theta_deg)
    wl_nm_list = np.round(wavelengths_um * 1000).astype(int)

    for i, wl_nm in enumerate(wl_nm_list):
        n_list = [1] + [nk_dict[m][i] for m in materials] + [1]
        res = coh_tmm(pol=pol, n_list=n_list, d_list=d_list, th_0=th0, lam_vac=wl_nm)
        R.append(res["R"])
        T.append(res["T"])
    R = np.asarray(R, dtype=np.float32)
    T = np.asarray(T, dtype=np.float32)
    return np.concatenate([R, T], axis=0)  # (2*N,)


# -------------------------
# 3) 结构 token 解析： "TiO2_158" -> (TiO2, 158)
# -------------------------
def parse_structure_tokens(tokens):
    mats = []
    thks = []
    for s in tokens:
        if "_" not in s:
            continue
        m, t = s.split("_", 1)
        mats.append(m)
        thks.append(float(t))
    return mats, thks


# -------------------------
# 4) Greedy decode（保持原版风格，但确保 src 形状正确）
# -------------------------
@torch.no_grad()
def greedy_decode(model, struc_index_dict, struc_word_dict, spec_target, max_len, start_symbol="BOS"):
    # 1) init ys = [BOS]
    bos_id = struc_word_dict[start_symbol]
    ys = torch.ones(1, 1).fill_(bos_id).long().to(DEVICE)

    # 2) src shape: (1,1,spec_dim)
    spec_np = np.asarray(spec_target, dtype=np.float32)  # (spec_dim,)
    src = torch.from_numpy(spec_np).to(DEVICE)[None, None, :]  # (1,1,spec_dim)
    src_mask = None

    out_tokens = []
    for _ in range(max_len - 1):
        trg_mask = Variable(subsequent_mask(ys.size(1)).type_as(src.data))
        out = model(src, Variable(ys), src_mask, trg_mask.to(DEVICE))

        prob = model.generator(out[:, -1])  # (1, vocab)
        _, next_word = torch.max(prob, dim=1)
        next_id = int(next_word.item())

        ys = torch.cat([ys, torch.ones(1, 1).long().to(DEVICE) * next_id], dim=1)

        sym = struc_index_dict.get(next_id, "UNK")
        if sym == "EOS":
            break
        out_tokens.append(sym)

    return out_tokens


# -------------------------
# 5) 主 eval：对 dev 集抽样/全量评估模型 MAE
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="model checkpoint .pt")
    ap.add_argument("--dev_struct", type=str, default="./dataset/Structure_dev.pkl")
    ap.add_argument("--dev_spec", type=str, default="./dataset/Spectrum_dev.pkl")
    ap.add_argument("--nk_dir", type=str, default="./dataset/data")
    ap.add_argument("--lambda0", type=float, default=0.9, help="wavelength start (um)")
    ap.add_argument("--lambda1", type=float, default=1.7, help="wavelength end (um)")
    ap.add_argument("--max_len", type=int, default=64, help="decode max length")
    ap.add_argument("--num_eval", type=int, default=200, help="how many dev samples to eval (<=0 means all)")
    ap.add_argument("--seed", type=int, default=0)
    args_cli = ap.parse_args()

    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    # ---- load ckpt ----
    ckpt = torch.load(args_cli.ckpt, map_location="cpu")
    cfg = ckpt["configs"]

    # ---- build model ----
    model = make_model_I(
        cfg.spec_dim,
        cfg.struc_dim,
        cfg.layers,
        cfg.d_model,
        cfg.d_ff,
        cfg.head_num,
        cfg.dropout,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---- load dicts (critical!) ----
    struc_word_dict = cfg.struc_word_dict
    struc_index_dict = cfg.struc_index_dict

    # ---- load dev data ----
    dev_struct = np.load(args_cli.dev_struct, allow_pickle=True)
    dev_spec = np.load(args_cli.dev_spec, allow_pickle=True)

    dev_spec = np.asarray(dev_spec, dtype=np.float32)
    N = len(dev_spec)
    spec_dim = dev_spec.shape[1]
    assert spec_dim == cfg.spec_dim, f"spec_dim mismatch: dev_spec={spec_dim}, cfg={cfg.spec_dim}"

    # ---- wavelengths from spec_dim ----
    n_pts = spec_dim // 2
    wavelengths_um = np.linspace(args_cli.lambda0, args_cli.lambda1, n_pts)

    # ---- materials needed from dict (粗暴但稳) ----
    # 只从词表里抓可能材料名：去掉特殊 token 和厚度
    mats = set()
    for k, idx in struc_word_dict.items():
        if "_" in k:
            mats.add(k.split("_")[0])
    mats = sorted(list(mats))
    nk_dict = load_nk(args_cli.nk_dir, mats, wavelengths_um)

    # ---- choose subset ----
    idxs = list(range(N))
    if args_cli.num_eval > 0 and args_cli.num_eval < N:
        idxs = random.sample(idxs, args_cli.num_eval)

    maes = []
    bad = 0

    for ii in idxs:
        spec_target = dev_spec[ii]  # (spec_dim,)
        # ---- decode ----
        pred_tokens = greedy_decode(
            model=model,
            struc_index_dict=struc_index_dict,
            struc_word_dict=struc_word_dict,
            spec_target=spec_target,
            max_len=args_cli.max_len,
            start_symbol="BOS",
        )

        # ---- parse ----
        mats_pred, thks_pred = parse_structure_tokens(pred_tokens)
        if len(mats_pred) == 0:
            bad += 1
            continue

        # ---- TMM -> spectrum ----
        try:
            spec_pred = calc_RT(mats_pred, thks_pred, nk_dict, wavelengths_um)
        except Exception:
            bad += 1
            continue

        mae = float(np.mean(np.abs(spec_pred - spec_target)))
        maes.append(mae)

    maes = np.asarray(maes, dtype=np.float32)
    print("==== DBR Eval (model decode MAE) ====")
    print(f"eval samples requested: {len(idxs)}")
    print(f"valid decoded samples : {len(maes)}")
    print(f"failed/empty          : {bad}")
    if len(maes) > 0:
        print(f"MAE mean  : {maes.mean():.6f}")
        print(f"MAE median: {np.median(maes):.6f}")
        print(f"MAE p90   : {np.quantile(maes, 0.90):.6f}")
        print(f"MAE p99   : {np.quantile(maes, 0.99):.6f}")


# if __name__ == "__main__":
#     main()
# python eval_dbr.py \
#   --ckpt saved_models/optogpt/in_paper/optogpt.pt \
#   --dev_struct ./dataset/Structure_dev.pkl \
#   --dev_spec ./dataset/Spectrum_dev.pkl \
#   --nk_dir ./dataset/data \
#   --lambda0 0.9 --lambda1 1.7 \
#   --max_len 64 \
#   --num_eval 200