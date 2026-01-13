#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_dbr_v5_full_debug.py

你要的“完整版 eval”，包含：
1) Oracle MAE：GT struct -> TMM -> spec vs dev_spec（定位数据一致性）
2) Pred MAE：greedy decode struct -> TMM -> spec vs dev_spec
3) Pred pair 分布统计（快速看是否坍塌）
4) 结构诊断：special/ood/不交替/厚度范围
5) 【实验一 Teacher-forcing】token-level NLL + top1 acc（排除 eval decode 的锅）
6) 【新增】GT token OOV ratio（定位“词表不一致/大量OOV”问题）
7) 额外打印：vocab size + special token id + 若干样例 token 是否在 vocab

用法示例：
python eval_dbr_v5_full_debug.py \
  --ckpt saved_models/optogpt/dbr_60k/model_inverse_best.pt \
  --dev_struct ./dataset/dbr/Structure_dev.pkl \
  --dev_spec   ./dataset/dbr/Spectrum_dev.pkl \
  --nk_dir     ./dataset/data \
  --lambda0 0.9 --lambda1 1.7 --step_um 0.005 \
  --max_len 64 \
  --num_eval 200 \
  --print_first_k 3 \
  --require_alternating
"""

import os
import argparse
import random
import pickle as pkl
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from scipy.interpolate import interp1d
from tmm import coh_tmm

# ====== optogpt 依赖 ======
from core.models.transformer import make_model_I, subsequent_mask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# utils
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
# 1) nk load
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
# 2) TMM calc (R,T concat)
# -------------------------
def calc_RT(materials, thicknesses_nm, nk_dict, wavelengths_um, pol="s", theta_deg=0.0):
    R, T = [], []
    d_list = [np.inf] + list(thicknesses_nm) + [np.inf]
    th0 = np.deg2rad(theta_deg)

    wl_nm_list = np.round(wavelengths_um * 1000).astype(int)  # match generator
    for i, wl_nm in enumerate(wl_nm_list):
        n_list = [1] + [nk_dict[m][i] for m in materials] + [1]
        res = coh_tmm(pol=pol, n_list=n_list, d_list=d_list, th_0=th0, lam_vac=int(wl_nm))
        R.append(res["R"])
        T.append(res["T"])

    R = np.asarray(R, dtype=np.float32)
    T = np.asarray(T, dtype=np.float32)
    return np.concatenate([R, T], axis=0)  # (2*N,)


# -------------------------
# 3) parse / pair
# -------------------------
def parse_structure_tokens(tokens):
    mats, thks = [], []
    for s in tokens:
        if "_" not in s:
            continue
        m, t = s.split("_", 1)
        try:
            mats.append(m)
            thks.append(float(t))
        except Exception:
            continue
    return mats, thks


def infer_pair_name_from_tokens(tokens):
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


# -------------------------
# 4) greedy decode (free-run)
# -------------------------
@torch.no_grad()
def greedy_decode(model, struc_index_dict, struc_word_dict, spec_target, max_len, start_symbol="BOS"):
    bos_id = struc_word_dict[start_symbol]
    ys = torch.ones(1, 1, dtype=torch.long, device=DEVICE).fill_(bos_id)

    spec_np = np.asarray(spec_target, dtype=np.float32)
    src = torch.from_numpy(spec_np).to(DEVICE)[None, None, :]  # (1,1,spec_dim)
    src_mask = None

    out_tokens = []
    for _ in range(max_len - 1):
        trg_mask = Variable(subsequent_mask(ys.size(1)).type_as(src.data)).to(DEVICE)
        out = model(src, Variable(ys), src_mask, trg_mask)
        prob = model.generator(out[:, -1])  # (1, vocab), log_softmax

        _, next_word = torch.max(prob, dim=1)  # greedy
        next_id = int(next_word.item())

        ys = torch.cat(
            [ys, torch.ones(1, 1, dtype=torch.long, device=DEVICE).fill_(next_id)],
            dim=1
        )

        sym = struc_index_dict.get(next_id, "UNK")
        if sym == "EOS":
            break
        out_tokens.append(sym)

    return out_tokens


# -------------------------
# 5) structure check
# -------------------------
SPECIAL = {"UNK", "PAD", "BOS", "EOS"}

def check_structure(tokens, allowed_materials=None, require_alternating=False):
    info = {}
    info["len_tokens"] = len(tokens)
    info["num_special"] = sum(1 for t in tokens if t in SPECIAL)
    info["has_special"] = info["num_special"] > 0

    mats, thks = parse_structure_tokens(tokens)
    info["num_layers_parsed"] = len(mats)

    if len(mats) == 0:
        info["reason"] = "empty_parsed"
        return False, info

    if len(mats) >= 2:
        info["is_alternating"] = all(mats[i] != mats[i - 1] for i in range(1, len(mats)))
    else:
        info["is_alternating"] = True

    if allowed_materials is not None:
        ood = [m for m in mats if m not in allowed_materials]
        info["num_ood_materials"] = len(ood)
        info["ood_materials_sample"] = sorted(list(set(ood)))[:5]
    else:
        info["num_ood_materials"] = 0
        info["ood_materials_sample"] = []

    thks_np = np.asarray(thks, dtype=np.float32)
    info["thk_min"] = float(thks_np.min())
    info["thk_max"] = float(thks_np.max())
    info["thk_mean"] = float(thks_np.mean())

    if info["has_special"]:
        info["reason"] = "contains_special_token"
        return False, info
    if info["num_ood_materials"] > 0:
        info["reason"] = "ood_material"
        return False, info
    if info["thk_min"] <= 0 or info["thk_max"] > 5000:
        info["reason"] = "bad_thickness_range"
        return False, info
    if require_alternating and not info.get("is_alternating", True):
        info["reason"] = "not_alternating"
        return False, info

    info["reason"] = "ok"
    return True, info


# -------------------------
# 6) teacher forcing experiment
# -------------------------
@torch.no_grad()
def teacher_forcing_metrics(model, struc_word_dict, gt_tokens, spec_target,
                            pad_symbol="PAD", bos_symbol="BOS", eos_symbol="EOS",
                            ignore_special=True):
    pad_id = struc_word_dict.get(pad_symbol, 0)
    bos_id = struc_word_dict.get(bos_symbol, None)
    eos_id = struc_word_dict.get(eos_symbol, None)
    unk_id = struc_word_dict.get("UNK", None)

    if bos_id is None:
        # 你的 cfg.struc_word_dict 应该有 BOS
        return 0.0, 0, 0.0, 0

    # map gt tokens to ids
    gt_ids = []
    for s in gt_tokens:
        if s in struc_word_dict:
            gt_ids.append(struc_word_dict[s])
        else:
            if unk_id is None:
                continue
            gt_ids.append(unk_id)

    if len(gt_ids) == 0:
        return 0.0, 0, 0.0, 0

    tgt_in = [bos_id] + gt_ids
    if eos_id is None:
        tgt_y = gt_ids + [pad_id]
    else:
        tgt_y = gt_ids + [eos_id]

    tgt_in_t = torch.tensor(tgt_in, dtype=torch.long, device=DEVICE)[None, :]  # (1,L)
    tgt_y_t  = torch.tensor(tgt_y,  dtype=torch.long, device=DEVICE)[None, :]  # (1,L)

    spec_np = np.asarray(spec_target, dtype=np.float32)
    src = torch.from_numpy(spec_np).to(DEVICE)[None, None, :]
    src_mask = None

    L = tgt_in_t.size(1)
    tgt_mask = subsequent_mask(L).type_as(src.data).to(DEVICE)

    out = model(src, tgt_in_t, src_mask, tgt_mask)  # (1,L,d)
    logp = model.generator(out)                      # (1,L,vocab) log_softmax

    nll = -logp.gather(-1, tgt_y_t.unsqueeze(-1)).squeeze(-1)  # (1,L)
    pred = logp.argmax(dim=-1)                                 # (1,L)
    correct = (pred == tgt_y_t).float()

    if ignore_special:
        ignore_ids = {pad_id, bos_id}
        if eos_id is not None:
            ignore_ids.add(eos_id)
        if unk_id is not None:
            ignore_ids.add(unk_id)
        mask = torch.ones_like(tgt_y_t, dtype=torch.bool)
        for iid in ignore_ids:
            mask &= (tgt_y_t != iid)
    else:
        mask = torch.ones_like(tgt_y_t, dtype=torch.bool)

    nll_sum = float(nll[mask].sum().item())
    n_tok = int(mask.sum().item())
    n_correct = float(correct[mask].sum().item())
    return nll_sum, n_tok, n_correct, n_tok


# -------------------------
# 7) OOV stats
# -------------------------
def compute_oov_ratio(dev_struct, struc_word_dict):
    tot = 0
    oov = 0
    for seq in dev_struct:
        for t in seq:
            tot += 1
            if t not in struc_word_dict:
                oov += 1
    return oov, tot, (oov / max(tot, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--dev_struct", type=str, default="./dataset/dbr/Structure_dev.pkl")
    ap.add_argument("--dev_spec", type=str, default="./dataset/dbr/Spectrum_dev.pkl")
    ap.add_argument("--nk_dir", type=str, default="./dataset/data")

    ap.add_argument("--lambda0", type=float, default=0.9)
    ap.add_argument("--lambda1", type=float, default=1.7)
    ap.add_argument("--step_um", type=float, default=0.005)

    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--num_eval", type=int, default=200, help="<=0 means all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--print_first_k", type=int, default=2)

    ap.add_argument("--require_alternating", action="store_true")
    ap.add_argument("--tf_include_special", action="store_true", help="if set, TF metrics include PAD/BOS/EOS/UNK")
    args = ap.parse_args()

    set_seed(args.seed)

    # ---- load ckpt ----
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["configs"]

    # ---- model ----
    model = make_model_I(
        cfg.spec_dim, cfg.struc_dim, cfg.layers, cfg.d_model, cfg.d_ff, cfg.head_num, cfg.dropout
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---- dicts ----
    struc_word_dict = cfg.struc_word_dict
    struc_index_dict = cfg.struc_index_dict

    print("\n[CKPT cfg]")
    print("  spec_dim:", cfg.spec_dim)
    print("  struc_dim:", cfg.struc_dim)
    print("  layers/d_model/d_ff/heads:", cfg.layers, cfg.d_model, cfg.d_ff, cfg.head_num)
    print("  dropout:", cfg.dropout)

    print("\n[Vocab sanity]")
    print("  vocab size:", len(struc_word_dict))
    for k in ["PAD", "BOS", "EOS", "UNK"]:
        if k in struc_word_dict:
            print(f"  {k:4s} id={struc_word_dict[k]}")
        else:
            print(f"  {k:4s} (missing in dict)")

    # ---- load dev ----
    dev_struct = load_pickle(args.dev_struct)
    dev_spec = load_pickle(args.dev_spec)
    dev_spec = np.asarray(dev_spec, dtype=np.float32)

    N = len(dev_spec)
    spec_dim = dev_spec.shape[1]
    assert spec_dim == cfg.spec_dim, f"spec_dim mismatch: dev_spec={spec_dim}, cfg={cfg.spec_dim}"

    # ---- wavelength grid ----
    n_pts = int(round((args.lambda1 - args.lambda0) / args.step_um)) + 1
    wavelengths_um = np.linspace(args.lambda0, args.lambda1, n_pts)
    wl_nm_list = np.round(wavelengths_um * 1000).astype(int)

    if spec_dim != 2 * n_pts:
        print("[FATAL] spec_dim != 2*n_pts")
        print("  spec_dim:", spec_dim)
        print("  n_pts:", n_pts, "=> 2*n_pts:", 2*n_pts)
        return

    print("\n[Dev data sanity]")
    print("  dev samples:", N)
    print("  dev_spec range: min=%.6f max=%.6f mean=%.6f" %
          (float(dev_spec.min()), float(dev_spec.max()), float(dev_spec.mean())))
    print("  wl_nm head/tail:", wl_nm_list[:5].tolist(), "...", wl_nm_list[-5:].tolist())

    # ---- OOV ratio ----
    oov, tot, ratio = compute_oov_ratio(dev_struct, struc_word_dict)
    print("\n[GT token OOV]")
    print(f"  OOV: {oov}/{tot} = {ratio:.4%}")
    if ratio > 0.01:
        print("  !! WARNING: OOV > 1%：强烈怀疑词表构建/保存/加载与 dev_struct 不一致。")

    # 打印几个 GT token 是否在 vocab
    print("\n[GT token examples in vocab?]")
    shown = 0
    for seq in dev_struct[:50]:
        for t in seq[:10]:
            print(" ", t, "->", ("IN" if t in struc_word_dict else "OOV"))
            shown += 1
            if shown >= 20:
                break
        if shown >= 20:
            break

    # ---- materials from dev_struct ----
    mats_set = set()
    for seq in dev_struct:
        for s in seq:
            if "_" in s:
                mats_set.add(s.split("_", 1)[0])
    mats = sorted(list(mats_set))
    print("\nmaterials(from dev_struct):", mats)

    nk_dict = load_nk(args.nk_dir, mats, wavelengths_um)

    # ---- choose subset ----
    idxs = list(range(N))
    if args.num_eval > 0 and args.num_eval < N:
        idxs = random.sample(idxs, args.num_eval)

    # ---- accumulators ----
    mae_oracle = []
    mae_pred = []
    bad_oracle = 0
    bad_pred = 0

    pred_pair_counter = Counter()

    stats = {
        "pred_ok": 0,
        "pred_has_special": 0,
        "pred_ood_material": 0,
        "pred_not_alternating": 0,
        "pred_empty": 0,
        "pred_bad_thk": 0,
        "pred_not_alternating_invalid": 0,
    }

    tf_nll_sum = 0.0
    tf_n_tok = 0
    tf_correct = 0.0

    ignore_special = (not args.tf_include_special)

    # ---- loop ----
    for j, ii in enumerate(idxs):
        spec_target = dev_spec[ii]
        gt_tokens = dev_struct[ii]

        # ---- Teacher forcing ----
        nll_sum, n_tok, n_cor, _ = teacher_forcing_metrics(
            model=model,
            struc_word_dict=struc_word_dict,
            gt_tokens=gt_tokens,
            spec_target=spec_target,
            ignore_special=ignore_special
        )
        tf_nll_sum += nll_sum
        tf_n_tok += n_tok
        tf_correct += n_cor

        # ---- Oracle MAE ----
        mats_gt, thks_gt = parse_structure_tokens(gt_tokens)
        if len(mats_gt) == 0:
            bad_oracle += 1
        else:
            try:
                spec_gt = calc_RT(mats_gt, thks_gt, nk_dict, wavelengths_um)
                mae_oracle.append(float(np.mean(np.abs(spec_gt - spec_target))))
            except Exception:
                bad_oracle += 1

        # ---- Pred (greedy) ----
        pred_tokens = greedy_decode(model, struc_index_dict, struc_word_dict, spec_target, args.max_len)
        pred_pair_counter[infer_pair_name_from_tokens(pred_tokens)] += 1

        ok, info = check_structure(
            pred_tokens, allowed_materials=set(mats), require_alternating=args.require_alternating
        )

        if not info.get("is_alternating", True):
            stats["pred_not_alternating"] += 1
            if args.require_alternating:
                stats["pred_not_alternating_invalid"] += 1

        if not ok:
            bad_pred += 1
            if info.get("reason") == "empty_parsed":
                stats["pred_empty"] += 1
            elif info.get("reason") == "contains_special_token":
                stats["pred_has_special"] += 1
            elif info.get("reason") == "ood_material":
                stats["pred_ood_material"] += 1
            elif info.get("reason") == "bad_thickness_range":
                stats["pred_bad_thk"] += 1
        else:
            stats["pred_ok"] += 1
            mats_pred, thks_pred = parse_structure_tokens(pred_tokens)
            try:
                spec_pred = calc_RT(mats_pred, thks_pred, nk_dict, wavelengths_um)
                mae_pred.append(float(np.mean(np.abs(spec_pred - spec_target))))
            except Exception:
                bad_pred += 1

        # ---- debug print ----
        if j < args.print_first_k:
            print("\n---- sample", ii, "----")
            print("target[min/max/mean]:",
                  float(spec_target.min()), float(spec_target.max()), float(spec_target.mean()))
            print("GT tokens head:", gt_tokens[:10], "... len=", len(gt_tokens))
            if len(mats_gt) > 0:
                alt = all(mats_gt[k] != mats_gt[k - 1] for k in range(1, len(mats_gt))) if len(mats_gt) > 1 else True
                print("GT pair:", infer_pair_name_from_tokens(gt_tokens), "| layers:", len(mats_gt), "| alt:", alt)
            print("PRED tokens head:", pred_tokens[:10], "... len=", len(pred_tokens))
            print("PRED pair:", infer_pair_name_from_tokens(pred_tokens))
            print("PRED check:", ok, info)
            if len(mae_oracle) > 0:
                print("oracle MAE(last):", mae_oracle[-1])
            if len(mae_pred) > 0:
                print("pred   MAE(last):", mae_pred[-1])
            if tf_n_tok > 0:
                print("TF running avg NLL:", tf_nll_sum / tf_n_tok,
                      "| TF running acc:", tf_correct / max(tf_n_tok, 1))

    # ---- report ----
    mae_oracle = np.asarray(mae_oracle, dtype=np.float32)
    mae_pred = np.asarray(mae_pred, dtype=np.float32)

    print("\n==================== REPORT ====================")
    print(f"eval requested samples: {len(idxs)}")

    print("\n[TEACHER FORCING] GT prefix -> next-token")
    print("  ignore_special:", ignore_special)
    print(f"  eval tokens: {tf_n_tok}")
    if tf_n_tok > 0:
        print(f"  avg NLL : {tf_nll_sum / tf_n_tok:.6f}")
        print(f"  top1 acc: {tf_correct / tf_n_tok:.4%}")
    else:
        print("  (no tokens evaluated; check mapping/OOV)")

    print("\n[ORACLE] GT structure -> TMM vs dev_spec")
    print(f"  valid oracle samples: {len(mae_oracle)}")
    print(f"  oracle failed       : {bad_oracle}")
    if len(mae_oracle) > 0:
        print(f"  oracle MAE mean  : {mae_oracle.mean():.6f}")
        print(f"  oracle MAE median: {np.median(mae_oracle):.6f}")
        print(f"  oracle MAE p90   : {np.quantile(mae_oracle, 0.90):.6f}")
        print(f"  oracle MAE p99   : {np.quantile(mae_oracle, 0.99):.6f}")

    print("\n[PRED] Greedy decode -> TMM vs dev_spec")
    print(f"  valid pred samples  : {len(mae_pred)}")
    print(f"  pred failed/invalid : {bad_pred}")
    if len(mae_pred) > 0:
        print(f"  pred MAE mean  : {mae_pred.mean():.6f}")
        print(f"  pred MAE median: {np.median(mae_pred):.6f}")
        print(f"  pred MAE p90   : {np.quantile(mae_pred, 0.90):.6f}")
        print(f"  pred MAE p99   : {np.quantile(mae_pred, 0.99):.6f}")

    print("\n[STRUCT DIAG]")
    for k, v in stats.items():
        print(f"  {k:28s}: {v}")

    print("\n[PRED pair distribution]")
    total_pred = sum(pred_pair_counter.values())
    for k, v in pred_pair_counter.most_common():
        print(f"  {k:20s} {v:6d} ({v / max(total_pred, 1):.3%})")

    print("\nHow to interpret:")
    print("1) Oracle MAE ~ 0：数据与TMM一致 ✅")
    print("2) TF acc 很低：模型/词表/目标定义有问题（不是 greedy decode 的锅）❌")
    print("3) 若 OOV 高：优先修词表构建/保存/加载一致性（PrepareData）")
    print("4) 若 OOV 低但 TF acc 仍低：token 粒度太细/长尾，建议厚度分桶或材料/厚度拆分")
    print("================================================")


if __name__ == "__main__":
    main()
