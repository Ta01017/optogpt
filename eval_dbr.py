import os
import numpy as np
import pickle as pkl
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from tmm import coh_tmm

from core.datasets.datasets import PrepareData
from core.models.transformer import make_model_I

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= 波段（与你生成器一致） =========
LAM_LOW, LAM_HIGH, LAM_STEP = 0.8, 1.7, 0.005
WAVELENGTHS = np.arange(LAM_LOW, LAM_HIGH, LAM_STEP)
WL_NM = (WAVELENGTHS * 1000).astype(int)
N_WL = len(WAVELENGTHS)

# ========= nk =========
def load_nk(materials, wavelengths, nk_dir):
    nk = {}
    for m in materials:
        df = np.loadtxt(os.path.join(nk_dir, f"{m}.csv"), delimiter=",", skiprows=1)
        wl, n, k = df[:,0], df[:,1], df[:,2]
        n_i = interp1d(wl, n, fill_value="extrapolate")(wavelengths)
        k_i = interp1d(wl, k, fill_value="extrapolate")(wavelengths)
        nk[m] = n_i + 1j * k_i
    return nk

# ========= token → 结构 =========
def parse_structure(tokens):
    mats, thks = [], []
    for t in tokens:
        if "_" not in t:
            continue
        m, d = t.split("_")
        mats.append(m)
        thks.append(float(d))
    return mats, thks

# ========= DBR 合法性检查 =========
def is_valid_dbr(mats):
    if len(mats) < 4 or len(mats) % 2 != 0:
        return False
    H, L = mats[0], mats[1]
    for i in range(len(mats)):
        if i % 2 == 0 and mats[i] != H:
            return False
        if i % 2 == 1 and mats[i] != L:
            return False
    return True

# ========= TMM =========
def simulate_rt(mats, thks, nk):
    R, T = [], []
    d_list = [np.inf] + thks + [np.inf]
    for i, wl in enumerate(WL_NM):
        n_list = [1] + [nk[m][i] for m in mats] + [1]
        res = coh_tmm("s", n_list, d_list, 0, wl)
        R.append(res["R"])
        T.append(res["T"])
    return np.array(R), np.array(T)

# ========= 主验证 =========
def eval_inverse(
    ckpt,
    train_struc, train_spec,
    dev_struc, dev_spec,
    nk_dir,
    spec_type="R_T",
    num_eval=50,
    plot=True
):
    data = PrepareData(
        train_struc, train_spec, 100,
        dev_struc, dev_spec,
        BATCH_SIZE=1,
        spec_type=spec_type,
        if_inverse="Inverse"
    )

    vocab = data.struc_word_dict
    idx2word = data.struc_index_dict
    BOS, EOS = vocab["BOS"], vocab["EOS"]

    spec_dim = len(data.dev_spec[0])
    model = make_model_I(
        spec_dim, len(vocab),
        layers=2, d_model=256, d_ff=1024,
        head_num=8, dropout=0.1
    ).to(DEVICE)

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE)["model_state_dict"])
    model.eval()

    # 收集 nk
    mats_used = set()
    with open(dev_struc, "rb") as f:
        for s in pkl.load(f):
            for t in s:
                mats_used.add(t.split("_")[0])
    mats_used.add("Glass_Substrate")
    nk = load_nk(mats_used, WAVELENGTHS, nk_dir)

    maes, valid = [], []

    for i in np.random.choice(len(data.dev_spec), num_eval, replace=False):
        src = torch.tensor(data.dev_spec[i]).float().to(DEVICE).unsqueeze(0)
        ys = torch.tensor([[BOS]], device=DEVICE)

        with torch.no_grad():
            mem = model.encode(src.unsqueeze(1), None)
            for _ in range(20):
                out = model.decode(mem, None, ys, None)
                nxt = model.generator(out[:,-1]).argmax(-1).item()
                ys = torch.cat([ys, torch.tensor([[nxt]], device=DEVICE)], 1)
                if nxt == EOS:
                    break

        tokens = [idx2word[j] for j in ys.squeeze().tolist()]
        mats, thks = parse_structure(tokens)

        ok = is_valid_dbr(mats)
        valid.append(ok)

        if not ok:
            continue

        R, T = simulate_rt(mats, thks, nk)
        pred = np.concatenate([R, T]) if spec_type in ["R_T","R+T"] else (R if spec_type=="R" else T)
        gt = np.array(data.dev_spec[i])

        mae = np.mean(np.abs(pred - gt))
        maes.append(mae)

        if plot:
            save_dir = "./eval_figs"
            os.makedirs(save_dir, exist_ok=True)

            plt.plot(gt, label="GT")
            plt.plot(pred, "--", label="Pred")
            plt.title(f"MAE={mae:.3f}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"case_{i}_mae_{mae:.3f}.png"))
            plt.close()

    print("====== DBR Validation ======")
    print(f"Valid DBR rate: {np.mean(valid):.3f}")
    print(f"MAE (valid only): {np.mean(maes):.4f}")



if __name__ == "__main__":
    eval_inverse(
        ckpt="saved_models/optogpt/dbr_60k_rt/model_inverse_R_T_best.pt",
        train_struc="./output/train_structure.pkl",
        train_spec="./output/train_spectrum.pkl",
        dev_struc="./output/dev_structure.pkl",
        dev_spec="./output/dev_spectrum.pkl",
        nk_dir="./data/nk/processed",
        spec_type="R_T",
        num_eval=50,
        plot=True
    )
