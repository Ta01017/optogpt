

# run_optogpt_dbr_60k_full.py
# 适配你新生成的 6w DBR 数据集（./dataset/Structure_{train,dev}.pkl & Spectrum_{train,dev}.pkl）
# 支持 spec_type: 'R' / 'T' / 'R_T'
# 修复：保存目录不存在的问题（自动创建）
# 建议配置：batch_size=256, warm_steps=8000, layers=2, spec_type=R_T

import os
import time
import argparse
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from core.datasets.datasets import PrepareData
from core.models.transformer import make_model_I
from core.trains.train import train_I, LabelSmoothing, NoamOpt, count_params


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def patch_saved_models_dir(save_root: str, save_folder: str):
    """
    你的 train.py 里通常硬编码：saved_models/optogpt/{save_folder}/...
    为了不改 train.py，这里提前把它创建出来，避免：
    RuntimeError: Parent directory saved_models/optogpt/test does not exist.
    """
    hardcoded = os.path.join('saved_models', 'optogpt', save_folder)
    ensure_dir(hardcoded)

    # 同时也创建你显式指定的 save_root/save_folder（如果你后续改 train.py 用它）
    ensure_dir(os.path.join(save_root, save_folder))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ===== Repro =====
    parser.add_argument('--seeds', default=42, type=int, help='random seeds')

    # ===== Train =====
    parser.add_argument('--epochs', default=500, type=int, help='Num of training epoches')
    parser.add_argument('--ratios', default=100, type=int, help='Ratio of training dataset (%)')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size (suggest 128/256)')
    parser.add_argument('--dropout', default=0.05, type=float, help='dropout rate')
    parser.add_argument('--max_lr', default=1e-3, type=float, help='NoamOpt factor (suggest 5e-4~1e-3)')
    parser.add_argument('--warm_steps', default=16000, type=int, help='NoamOpt warmup steps (suggest 5k~12k)')

    parser.add_argument('--smoothing', default=0.1, type=float, help='Label smoothing for KL')

    # ===== Model =====
    parser.add_argument('--layers', default=1, type=int, help='Encoder layers (DBR suggest 2)')
    parser.add_argument('--head_num', default=8, type=int, help='Attention head numbers')
    parser.add_argument('--d_model', default=256, type=int, help='Attention model dim')
    parser.add_argument('--d_ff', default=1024, type=int, help='FFN dim')
    parser.add_argument('--max_len', default=22, type=int, help='Max token length (DBR 20 layers + BOS/EOS)')

    # ===== Save =====
    parser.add_argument('--save_root', default='saved_models/optogpt', type=str, help='Root folder (for future use)')
    parser.add_argument('--save_folder', default='dbr_60k', type=str, help='Subfolder under saved_models/optogpt/')
    parser.add_argument('--save_name', default='model_inverse', type=str, help='Model name prefix')

    # ===== Spec Type =====
    # 你的数据存的是 [R..., T...] 拼接，训练时按 spec_type 切：
    # - 'R'   : 前半
    # - 'T'   : 后半
    # - 'R_T' : 全部（不切）
    parser.add_argument('--spec_type', default='R_T', type=str, help='R / T / R_T')

    # ===== Data (6w split pkl) =====
    parser.add_argument('--TRAIN_FILE', default='./dataset/dbr/Structure_train.pkl', type=str)
    parser.add_argument('--TRAIN_SPEC_FILE', default='./dataset/dbr/Spectrum_train.pkl', type=str)
    parser.add_argument('--DEV_FILE', default='./dataset/dbr/Structure_dev.pkl', type=str)
    parser.add_argument('--DEV_SPEC_FILE', default='./dataset/dbr/Spectrum_dev.pkl', type=str)

    # ===== Dict placeholders (will be overwritten by PrepareData) =====
    parser.add_argument('--struc_index_dict', default={2: 'BOS'}, type=dict)
    parser.add_argument('--struc_word_dict', default={'BOS': 2}, type=dict)

    args = parser.parse_args()

    # ---- seeds ----
    torch.manual_seed(args.seeds)
    np.random.seed(args.seeds)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seeds)

    # ---- create save dirs to avoid runtime error ----
    patch_saved_models_dir(args.save_root, args.save_folder)

    # ---- naming ----
    temp = [
        args.ratios, args.smoothing, args.batch_size,
        args.max_lr, args.warm_steps,
        args.layers, args.head_num, args.d_model, args.d_ff
    ]
    args.save_name = f"{args.save_name}_{args.spec_type}_S_R_B_LR_WU_L_H_D_F_{temp}"

    # ---- load data ----
    # 注意：你的 PrepareData 内部需要：
    # 1) spec = np.array(spec, dtype=np.float32) 才能有 shape
    # 2) 支持 spec_type='R_T'（不切）
    data = PrepareData(
        args.TRAIN_FILE,
        args.TRAIN_SPEC_FILE,
        args.ratios,
        args.DEV_FILE,
        args.DEV_SPEC_FILE,
        args.batch_size,
        args.spec_type,
        'Inverse'
    )

    tgt_vocab = len(data.struc_word_dict)   # structure vocab size
    spec_dim = len(data.dev_spec[0])        # spectrum dim after slicing

    args.struc_dim = tgt_vocab
    args.spec_dim = spec_dim
    args.struc_index_dict = data.struc_index_dict
    args.struc_word_dict = data.struc_word_dict

    print(f"[Data] spec_dim={args.spec_dim} | struc_vocab={args.struc_dim}")
    print(f"[Data] train_batches={len(data.train_data)} | dev_batches={len(data.dev_data)}")
    print(f"[Save] folder=saved_models/optogpt/{args.save_folder} | name={args.save_name}")

    # ---- model ----
    model = make_model_I(
        args.spec_dim,
        args.struc_dim,
        args.layers,
        args.d_model,
        args.d_ff,
        args.head_num,
        args.dropout
    ).to(DEVICE)

    print('Model Transformer, Number of parameters {}'.format(count_params(model)))

    # ---- train ----
    print(">>>>>>> start train")
    train_start = time.time()

    criterion = LabelSmoothing(tgt_vocab, padding_idx=0, smoothing=args.smoothing)
    optimizer = NoamOpt(
        args.d_model,
        args.max_lr,
        args.warm_steps,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    )

    train_I(data, model, criterion, optimizer, args, DEVICE)
    print(f"<<<<<<< finished train, cost {time.time() - train_start:.2f} seconds")
