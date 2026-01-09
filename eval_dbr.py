#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_dbr.py
- 支持 spec_type: "R", "T", "R_T"/"R+T"
- 两个改动已集成：
  1) min_len_layers 前禁止生成 EOS（避免 layers=0 collapse）
  2) 简单 DBR grammar（奇数层从 Hset 选，偶数层从 Lset 选；只允许 Mat_thick token）

运行：python eval_dbr.py
（参数都写在 main_config() 里，不用命令行）
"""

import os
import re
import copy
import math
import random
import pickle as pkl
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tmm import coh_tmm

# ====== 你的工程 import（按你现有结构）======
from core.datasets.datasets import PrepareData, Batch, PAD
from core.models.transformer import make_model_I

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# 0) 配置（你不想写命令行就写这里）
# -----------------------------
@dataclass
class Cfg:
    ckpt: str = "saved_models/optogpt/test/model_inverse_R_best.pt"

    train_struc: str = "./output/train_structure.pkl"
    train_spec: str  = "./output/train_spectrum.pkl"
    dev_struc: str   = "./output/dev_structure.pkl"
    dev_spec: str    = "./output/dev_spectrum.pkl"

    nk_dir: str = "./data/nk/processed"

    spec_type: str = "R_T"   # "R" / "T" / "R_T" / "R+T"
    num_eval: int = 50

    # 你的数据生成器波段（必须一致）
    lam_low: float = 0.8
    lam_high: float = 1.7
    lam_step: float = 0.005

    # decode 超参
    max_len_layers: int = 22
    min_len_layers: int = 4
    temperature: float = 1.0

    # 模型结构（必须和训练一致）
    N: int = 2
    d_model: int = 256
    d_ff: int = 1024
    h: int = 8
    dropout: float = 0.1

    # 语法分组参数（按 n 中位数分H/L）
    dn_th: float = 0.15
    k_max_th: float = 1e-3

    # 保存图像
    save_fig: bool = True
    fig_dir: str = "./eval_figs"


def main_config() -> Cfg:
    cfg = Cfg()
    os.makedirs(cfg.fig_dir, exist_ok=True)
    return cfg


# -----------------------------
# 1) nk 读取（csv: wl,n,k）
# -----------------------------
def load_nk(materials: List[str], wavelengths: np.ndarray, nk_dir: str) -> Dict[str, np.ndarray]:
    nk = {}
    for m in materials:
        path = os.path.join(nk_dir, f"{m}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"nk file not found: {path}")
        df = pd.read_csv(path)
        wl = df.iloc[:, 0].to_numpy()
        n  = df.iloc[:, 1].to_numpy()
        k  = df.iloc[:, 2].to_numpy()
        n_i = interp1d(wl, n, bounds_error=False, fill_value="extrapolate")(wavelengths)
        k_i = interp1d(wl, k, bounds_error=False, fill_value="extrapolate")(wavelengths)
        nk[m] = n_i + 1j * k_i
    return nk


# -----------------------------
# 2) 词表解析：只允许 Mat_Thick token
# -----------------------------
def build_token_sets(vocab: Dict[str, int]):
    word2idx = vocab
    idx2word = {v: k for k, v in word2idx.items()}

    struct_ids: List[int] = []
    mat_of_id: Dict[int, str] = {}
    thick_of_id: Dict[int, int] = {}

    for wid, w in idx2word.items():
        if "_" not in w:
            continue
        mat, thick = w.rsplit("_", 1)
        if thick.isdigit():
            struct_ids.append(wid)
            mat_of_id[wid] = mat
            thick_of_id[wid] = int(thick)

    return idx2word, struct_ids, mat_of_id, thick_of_id


def parse_structure(tokens: List[str]) -> Tuple[List[str], List[float]]:
    mats, thks = [], []
    for t in tokens:
        if "_" not in t:
            continue
        mat, d = t.rsplit("_", 1)
        if d.isdigit():
            mats.append(mat)
            thks.append(float(d))
    return mats, thks


# -----------------------------
# 3) H/L 集合（简单 grammar）
# -----------------------------
def pick_HL_sets_by_n(
    nk_dict: Dict[str, np.ndarray],
    wavelengths: np.ndarray,
    candidates_mats: List[str],
    k_max_th: float = 1e-3,
    dn_th: float = 0.15,
) -> Tuple[Set[str], Set[str]]:
    i_mid = len(wavelengths) // 2
    n_med = {}
    for m in candidates_mats:
        if m not in nk_dict:
            continue
        n_med[m] = float(np.real(nk_dict[m][i_mid]))

    global_med = float(np.median(list(n_med.values()))) if len(n_med) else 1.5

    Hset, Lset = set(), set()
    for m, n in n_med.items():
        kmax = float(np.max(np.abs(np.imag(nk_dict[m])))) if m in nk_dict else 0.0
        if kmax > k_max_th:
            continue
        if n >= global_med + dn_th:
            Hset.add(m)
        elif n <= global_med - dn_th:
            Lset.add(m)

    # 兜底：top/bottom 30%
    if len(Hset) < 2 or len(Lset) < 2:
        mats_sorted = sorted(n_med.items(), key=lambda kv: kv[1])
        n = len(mats_sorted)
        cut = max(1, int(0.3 * n))
        Lset = set([m for m, _ in mats_sorted[:cut]])
        Hset = set([m for m, _ in mats_sorted[-cut:]])

    return Hset, Lset


def allowed_ids_for_position(
    pos_layer_1based: int,
    struct_ids: List[int],
    mat_of_id: Dict[int, str],
    Hset: Set[str],
    Lset: Set[str],
) -> List[int]:
    use_set = Hset if (pos_layer_1based % 2 == 1) else Lset
    out = []
    for wid in struct_ids:
        m = mat_of_id.get(wid, None)
        if m in use_set:
            out.append(wid)
    return out


# -----------------------------
# 4) TMM（coh_tmm）算 R/T
# -----------------------------
def simulate_rt(mats: List[str], thks: List[float], nk: Dict[str, np.ndarray], wl_nm: np.ndarray):
    if len(mats) == 0:
        # 空结构：给一个“全透”近似（避免崩）
        R = np.zeros_like(wl_nm, dtype=np.float32)
        T = np.ones_like(wl_nm, dtype=np.float32)
        return R, T

    R, T = [], []
    d_list = [np.inf] + thks + [np.inf]
    for i, wl in enumerate(wl_nm):
        n_list = [1] + [nk[m][i] for m in mats] + [1]
        res = coh_tmm("s", n_list, d_list, 0, float(wl))
        R.append(res["R"])
        T.append(res["T"])
    return np.array(R, dtype=np.float32), np.array(T, dtype=np.float32)


# -----------------------------
# 5) Decode：min_len EOS + H/L grammar
# -----------------------------
def decode_with_minlen_and_grammar(
    model,
    src_spec: torch.Tensor,          # [1, spec_dim]
    BOS: int,
    EOS: int,
    struct_ids: List[int],
    mat_of_id: Dict[int, str],
    Hset: Set[str],
    Lset: Set[str],
    min_len_layers: int,
    max_len_layers: int,
    temperature: float,
) -> List[int]:
    # Transformer_I：memory = fc(src)
    with torch.no_grad():
        memory = model.fc(src_spec)  # [1, d_model]

    ys = torch.tensor([[BOS]], device=src_spec.device, dtype=torch.long)

    for _ in range(max_len_layers):
        tgt_mask = Batch.make_std_mask(ys, PAD).to(src_spec.device)

        with torch.no_grad():
            out = model.decode(memory, None, ys, tgt_mask)     # [1, len, d_model]
            logp = model.generator(out[:, -1])                 # [1, vocab] (log_softmax)

        gen_layers = ys.size(1) - 1  # 不含 BOS
        # 1) min_len 前禁止 EOS
        if gen_layers < min_len_layers:
            logp[:, EOS] = -1e9

        # 2) H/L grammar：第 (gen_layers+1) 层决定允许集合
        pos_layer_1based = gen_layers + 1
        allowed = allowed_ids_for_position(pos_layer_1based, struct_ids, mat_of_id, Hset, Lset)

        # 超过 min_len 才允许 EOS
        if gen_layers >= min_len_layers:
            allowed = allowed + [EOS]

        # mask 只允许 allowed
        masked = torch.full_like(logp, -1e9)
        masked[:, allowed] = logp[:, allowed]
        logp = masked

        if temperature != 1.0:
            logp = logp / temperature

        nxt = int(torch.argmax(logp, dim=-1).item())
        ys = torch.cat([ys, torch.tensor([[nxt]], device=src_spec.device, dtype=torch.long)], dim=1)

        if nxt == EOS:
            break

    return ys.squeeze(0).tolist()


# -----------------------------
# 6) spec_type 对齐 + MAE
# -----------------------------
def build_pred_spec(R: np.ndarray, T: np.ndarray, spec_type: str) -> np.ndarray:
    if spec_type == "R":
        return R
    if spec_type == "T":
        return T
    if spec_type in ["R_T", "R+T"]:
        return np.concatenate([R, T], axis=0)
    raise ValueError(f"Unknown spec_type: {spec_type}")


def plot_and_save(gt: np.ndarray, pred: np.ndarray, save_path: str, title: str):
    plt.figure(figsize=(6, 4))
    plt.plot(gt, label="GT")
    plt.plot(pred, "--", label="Pred")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# -----------------------------
# 7) 主流程
# -----------------------------
def main(cfg: Cfg):
    print("====== DBR Validation ======")
    print(f"Spec type: {cfg.spec_type}")
    print(f"Device: {DEVICE}")

    # 波段网格（与生成器一致）
    wavelengths = np.arange(cfg.lam_low, cfg.lam_high, cfg.lam_step)  # 不含 lam_high
    wl_nm = (wavelengths * 1000).astype(int)
    n_wl = len(wavelengths)

    # 读数据（PrepareData 会按 spec_type 切分/不切分）
    data = PrepareData(
        cfg.train_struc, cfg.train_spec, 100,
        cfg.dev_struc, cfg.dev_spec,
        BATCH_SIZE=1,
        spec_type=cfg.spec_type,
        if_inverse="Inverse",
    )

    vocab = data.struc_word_dict
    idx2word = data.struc_index_dict
    BOS = vocab["BOS"]
    EOS = vocab["EOS"]

    # 确认维度一致
    spec_dim = len(data.dev_spec[0])
    if cfg.spec_type in ["R_T", "R+T"]:
        assert spec_dim == 2 * n_wl, f"Spec dim mismatch: data={spec_dim}, expected={2*n_wl}"
    else:
        assert spec_dim == n_wl, f"Spec dim mismatch: data={spec_dim}, expected={n_wl}"

    # 从词表抽结构 token
    idx2word_full, struct_ids, mat_of_id, thick_of_id = build_token_sets(vocab)

    # 收集材料集合：从词表里的材料 + 你nk目录存在的
    candidates_mats = sorted(list(set(mat_of_id.values())))

    # 读 nk（只读用到的材料）
    nk = load_nk(candidates_mats, wavelengths, cfg.nk_dir)

    # H/L 分组
    Hset, Lset = pick_HL_sets_by_n(nk, wavelengths, candidates_mats, cfg.k_max_th, cfg.dn_th)
    print(f"[Grammar] Hset size={len(Hset)}, Lset size={len(Lset)}")

    # 构造模型 & load ckpt
    model = make_model_I(
        src_vocab=spec_dim,
        tgt_vocab=len(vocab),
        N=cfg.N,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        h=cfg.h,
        dropout=cfg.dropout
    ).to(DEVICE)

    ckpt = torch.load(cfg.ckpt, map_location=DEVICE)
    # 兼容不同保存字段
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # 评估抽样
    devN = len(data.dev_spec)
    picks = np.random.choice(devN, min(cfg.num_eval, devN), replace=False)

    maes_all = []
    lengths = []

    for j, idx in enumerate(picks, 1):
        gt_spec = np.array(data.dev_spec[idx], dtype=np.float32)

        src = torch.tensor(gt_spec, device=DEVICE).float().unsqueeze(0)

        ids = decode_with_minlen_and_grammar(
            model=model,
            src_spec=src,
            BOS=BOS,
            EOS=EOS,
            struct_ids=struct_ids,
            mat_of_id=mat_of_id,
            Hset=Hset,
            Lset=Lset,
            min_len_layers=cfg.min_len_layers,
            max_len_layers=cfg.max_len_layers,
            temperature=cfg.temperature
        )

        tokens = [idx2word[i] for i in ids if i in idx2word]
        mats, thks = parse_structure(tokens)
        lengths.append(len(mats))

        R, T = simulate_rt(mats, thks, nk, wl_nm)
        pred_spec = build_pred_spec(R, T, cfg.spec_type)

        if pred_spec.shape != gt_spec.shape:
            print(f"[{j}/{len(picks)}] idx={idx} SHAPE MISMATCH pred={pred_spec.shape} gt={gt_spec.shape}")
            continue

        mae = float(np.mean(np.abs(pred_spec - gt_spec)))
        maes_all.append(mae)

        if cfg.save_fig:
            save_path = os.path.join(cfg.fig_dir, f"case_{idx:06d}_mae_{mae:.4f}.png")
            title = f"idx={idx} | layers={len(mats)} | MAE={mae:.4f}"
            plot_and_save(gt_spec, pred_spec, save_path, title)
            print(f"[{j}/{len(picks)}] idx={idx} OK, layers={len(mats)}, MAE={mae:.4f}, saved={save_path}")
        else:
            print(f"[{j}/{len(picks)}] idx={idx} OK, layers={len(mats)}, MAE={mae:.4f}")

    print("\n====== Summary ======")
    if len(maes_all) == 0:
        print("No valid eval cases (all mismatched).")
        return
    print(f"MAE (all): {np.mean(maes_all):.4f}  (n={len(maes_all)})")
    print(f"Length mean: {np.mean(lengths):.2f} | min={np.min(lengths)} max={np.max(lengths)}")
    print(f"Figures saved to: {cfg.fig_dir}")


if __name__ == "__main__":
    cfg = main_config()
    main(cfg)
