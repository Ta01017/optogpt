# eval_dbr.py
import os
import argparse
import numpy as np
import pickle as pkl
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tmm import coh_tmm
from torch.autograd import Variable

# ====== 导入你项目里的数据处理和模型构建 ======
from core.datasets.datasets import PrepareData
from core.models.transformer import make_model_I   # 你贴出来的 make_model_I

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_ID = 1  # 你的 datasets.py 里 PAD=1


# -----------------------------
# masks (与项目保持一致)
# -----------------------------
def subsequent_mask(size: int):
    attn_shape = (1, size, size)
    subsequent = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(subsequent) == 0

def make_std_mask(tgt: torch.Tensor, pad: int = PAD_ID):
    # tgt: (B, L)
    tgt_mask = (tgt != pad).unsqueeze(-2)  # (B, 1, L)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask


# -----------------------------
# nk 加载（支持 alias）
# -----------------------------
NAME_ALIAS = {
    "Glass": "Glass_Substrate",
}

def load_nk(materials, wavelengths_um, nk_dir):
    nk = {}
    for m in materials:
        m_file = NAME_ALIAS.get(m, m)
        path = os.path.join(nk_dir, f"{m_file}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"nk file not found: {path}")

        # csv: wl,n,k  (wl 单位必须与 wavelengths_um 一致：这里用 um)
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        wl = data[:, 0]
        n = data[:, 1]
        k = data[:, 2]
        n_i = interp1d(wl, n, bounds_error=False, fill_value="extrapolate")(wavelengths_um)
        k_i = interp1d(wl, k, bounds_error=False, fill_value="extrapolate")(wavelengths_um)
        nk[m] = n_i + 1j * k_i
    return nk


# -----------------------------
# token -> materials, thicknesses
# -----------------------------
def parse_structure(tokens):
    mats, thks = [], []
    for t in tokens:
        if t in ("BOS", "EOS", "PAD", "UNK"):
            continue
        if "_" not in t:
            continue
        m, d = t.split("_", 1)
        try:
            mats.append(m)
            thks.append(float(d))
        except Exception:
            continue
    return mats, thks


# -----------------------------
# DBR 合法性检查（HLHL...）
# -----------------------------
def is_valid_dbr(mats):
    if len(mats) < 4 or (len(mats) % 2) != 0:
        return False
    H, L = mats[0], mats[1]
    for i, m in enumerate(mats):
        if (i % 2 == 0 and m != H) or (i % 2 == 1 and m != L):
            return False
    return True


# -----------------------------
# TMM: 计算 R,T
# - 这里默认：空气 | stack | 空气（与你生成器的 coh_tmm 写法一致）
# - 如果你生成时是空气 | stack | Glass_Substrate，请改 use_substrate=True
# -----------------------------
def simulate_rt(mats, thks_nm, nk, wl_nm, use_substrate=False, substrate_name="Glass_Substrate"):
    R, T = [], []
    d_list = [np.inf] + thks_nm + [np.inf]

    for i, lam in enumerate(wl_nm):
        if use_substrate:
            n_list = [1] + [nk[m][i] for m in mats] + [nk[substrate_name][i]]
        else:
            n_list = [1] + [nk[m][i] for m in mats] + [1]

        res = coh_tmm("s", n_list, d_list, 0, lam)
        R.append(res["R"])
        T.append(res["T"])
    return np.array(R, dtype=np.float32), np.array(T, dtype=np.float32)


# -----------------------------
# greedy decode (用 model.fc + model.decode)
# -----------------------------
def greedy_decode_inverse(model, src_spec_vec, BOS, EOS, max_len=22):
    """
    src_spec_vec: torch.FloatTensor shape (1, spec_dim)
    Return: token id list (including BOS..EOS)
    """
    model.eval()
    ys = torch.tensor([[BOS]], device=DEVICE, dtype=torch.long)  # (1,1)

    with torch.no_grad():
        # memory = fc(src)  (src 需要是 (B, spec_dim) 或 (B, 1, spec_dim) 取决于训练时喂的形状)
        # 你 PrepareData 里 Inverse: self.src = trg.unsqueeze(-2)
        # 即 src 形状是 (B, 1, spec_dim)，所以这里也对齐：unsqueeze(1)
        memory = model.fc(src_spec_vec.unsqueeze(1))  # (1, 1, d_model)

        for _ in range(max_len - 1):
            tgt_mask = make_std_mask(ys, pad=PAD_ID)      # (1,1,L)
            out = model.decode(memory, None, ys, tgt_mask)  # (1, L, d_model)
            logp = model.generator(out[:, -1])            # (1, vocab)
            next_id = torch.argmax(logp, dim=-1).item()
            ys = torch.cat([ys, torch.tensor([[next_id]], device=DEVICE)], dim=1)
            if next_id == EOS:
                break

    return ys.squeeze(0).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="path to saved .pt checkpoint")
    parser.add_argument("--train_struc", type=str, required=True, help="train_structure.pkl")
    parser.add_argument("--train_spec", type=str, required=True, help="train_spectrum.pkl")
    parser.add_argument("--dev_struc", type=str, required=True, help="dev_structure.pkl")
    parser.add_argument("--dev_spec", type=str, required=True, help="dev_spectrum.pkl")
    parser.add_argument("--nk_dir", type=str, required=True, help="directory containing nk csv files")
    parser.add_argument("--spec_type", type=str, default="R_T", choices=["R", "T", "R_T", "R+T"])
    parser.add_argument("--num_eval", type=int, default=50)
    parser.add_argument("--max_len", type=int, default=22)
    parser.add_argument("--save_dir", type=str, default="./eval_figs")
    parser.add_argument("--use_substrate", action="store_true", help="use air|stack|Glass_Substrate instead of air|stack|air")
    parser.add_argument("--substrate_name", type=str, default="Glass_Substrate")

    # 波段（必须与你生成器一致）
    parser.add_argument("--lam_low", type=float, default=0.8)
    parser.add_argument("--lam_high", type=float, default=1.7)
    parser.add_argument("--lam_step", type=float, default=0.005)

    # 模型超参（必须与你训练时一致）
    parser.add_argument("--N", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--h", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ===== 波长 grid =====
    wavelengths = np.arange(args.lam_low, args.lam_high, args.lam_step)  # um
    wl_nm = (wavelengths * 1000).astype(int)
    n_wl = len(wavelengths)

    # ===== PrepareData（确保 spec_type 行为与你 datasets.py 修改一致）=====
    data = PrepareData(
        args.train_struc, args.train_spec, 100,
        args.dev_struc, args.dev_spec,
        BATCH_SIZE=1,
        spec_type=args.spec_type,
        if_inverse="Inverse"
    )

    vocab = data.struc_word_dict
    idx2word = data.struc_index_dict
    BOS = vocab["BOS"]
    EOS = vocab["EOS"]

    spec_dim = len(data.dev_spec[0])
    tgt_vocab = len(vocab)

    # ===== 构建模型并加载 ckpt =====
    model = make_model_I(
        spec_dim, tgt_vocab,
        args.N, args.d_model, args.d_ff, args.h, args.dropout
    ).to(DEVICE)

    ckpt = torch.load(args.ckpt, map_location=DEVICE)
    # 兼容不同保存字段名
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    # ===== 收集 dev 里用到的材料，加载 nk =====
    mats_used = set()
    with open(args.dev_struc, "rb") as f:
        dev_structs = pkl.load(f)
        for s in dev_structs:
            for t in s:
                if "_" in t:
                    mats_used.add(t.split("_", 1)[0])
    if args.use_substrate:
        mats_used.add(args.substrate_name)

    nk = load_nk(sorted(list(mats_used)), wavelengths, args.nk_dir)

    # ===== 评估 =====
    rng = np.random.default_rng(42)
    eval_indices = rng.choice(len(data.dev_spec), size=min(args.num_eval, len(data.dev_spec)), replace=False)

    valid_flags = []
    maes = []

    for k, i in enumerate(eval_indices, 1):
        # dev_spec[i] 是 numpy array
        src_vec = torch.tensor(data.dev_spec[i], dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1, spec_dim)

        # greedy decode
        pred_ids = greedy_decode_inverse(model, src_vec, BOS, EOS, max_len=args.max_len)
        pred_tokens = [idx2word[j] for j in pred_ids]

        mats, thks = parse_structure(pred_tokens)
        ok = is_valid_dbr(mats)
        valid_flags.append(ok)

        if not ok:
            print(f"[{k}/{len(eval_indices)}] idx={i} invalid DBR, tokens(head)={pred_tokens[:8]}")
            continue

        # 物理回算
        R, T = simulate_rt(mats, thks, nk, wl_nm, use_substrate=args.use_substrate, substrate_name=args.substrate_name)

        if args.spec_type in ["R_T", "R+T"]:
            pred_spec = np.concatenate([R, T])
        elif args.spec_type == "R":
            pred_spec = R
        else:
            pred_spec = T

        gt_spec = np.array(data.dev_spec[i], dtype=np.float32)
        mae = float(np.mean(np.abs(pred_spec - gt_spec)))
        maes.append(mae)

        # 保存图（服务器无 show，用 savefig）
        fig_path = os.path.join(args.save_dir, f"case_{i:06d}_mae_{mae:.4f}.png")
        plt.figure(figsize=(7, 3))
        plt.plot(gt_spec, label="GT", linewidth=2)
        plt.plot(pred_spec, "--", label="ReSim (from pred structure)", linewidth=2)
        plt.title(f"idx={i} | MAE={mae:.4f} | layers={len(mats)}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()

        print(f"[{k}/{len(eval_indices)}] idx={i} OK, layers={len(mats)}, MAE={mae:.4f}, saved={fig_path}")

    valid_rate = float(np.mean(valid_flags)) if len(valid_flags) else 0.0
    mae_mean = float(np.mean(maes)) if len(maes) else float("nan")

    print("\n====== DBR Validation ======")
    print(f"Spec type: {args.spec_type}")
    print(f"Valid DBR rate: {valid_rate:.3f} ({sum(valid_flags)}/{len(valid_flags)})")
    print(f"MAE (valid only): {mae_mean:.6f}  (n={len(maes)})")
    print(f"Figures saved to: {args.save_dir}")


if __name__ == "__main__":

    class Args:
        # ===== checkpoint =====
        ckpt = "saved_models/optogpt/test/model_inverse_R_best.pt"

        # ===== dataset =====
        train_struc = "./output/train_structure.pkl"
        train_spec  = "./output/train_spectrum.pkl"
        dev_struc   = "./output/dev_structure.pkl"
        dev_spec    = "./output/dev_spectrum.pkl"

        # ===== nk =====
        nk_dir = "./data/nk/processed"

        # ===== spectrum type =====
        spec_type = "R"          # R / T / R_T / R+T

        # ===== eval control =====
        num_eval = 50
        max_len = 22
        save_dir = "./eval_figs"

        # ===== optical setting (必须和生成器一致) =====
        lam_low  = 0.8
        lam_high = 1.7
        lam_step = 0.005

        # ===== model hyper-params (必须和训练一致) =====
        N = 2
        d_model = 256
        d_ff = 1024
        h = 8
        dropout = 0.1

        # ===== substrate =====
        use_substrate = False
        substrate_name = "Glass_Substrate"

    args = Args()

    # 👉 直接调用 main 逻辑（把 main() 稍微改成接收 args）
    main(args)

