#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from tqdm import tqdm

def build_wl95(wl0=900.0, wl1=1670.0, target_bands_115=115, crop_left=15, crop_right=5):
    wl115 = np.linspace(wl0, wl1, target_bands_115, dtype=np.float32)
    wl95  = wl115[crop_left: target_bands_115 - crop_right]
    return wl95

def select_115_then_crop95(image_hwc: np.ndarray, target_bands_115=115, crop_left=15, crop_right=5):
    """
    完全复刻你 dataset 里的逻辑：
    total_bands -> 均匀抽到 115 -> 再裁成 95（15:-5）
    """
    total_bands = image_hwc.shape[2]
    step = total_bands / target_bands_115
    selected_bands = [int(i * step) for i in range(target_bands_115)]
    img115 = image_hwc[:, :, selected_bands]
    img95  = img115[:, :, crop_left: target_bands_115 - crop_right]
    assert img95.shape[2] == 95
    return img95

def apply_W(image_hwc_95: np.ndarray, W_6x95: np.ndarray):
    """
    image_hwc_95: [H,W,95]
    W_6x95:       [6,95]
    out:          [H,W,6]
    """
    # [H,W,95] @ [95,6] => [H,W,6]
    out = image_hwc_95.astype(np.float32) @ W_6x95.T.astype(np.float32)
    return out.astype(np.float32)

def main(
    root_dir: str,
    w_npz: str = "W_6x95_from_excel_T.npz",
    in_subdir: str = "visible",
    out_subdir: str = "visible_w6",
    use_key: str = "W",   # "W" 或 "W_raw"（通常用 W）
    overwrite: bool = False,
):
    z = np.load(w_npz)
    W = z[use_key].astype(np.float32)   # [6,95]
    assert W.shape == (6,95), f"W shape must be (6,95), got {W.shape}"

    in_dir  = os.path.join(root_dir, in_subdir)
    out_dir = os.path.join(root_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(in_dir, "*.npy")))
    print(f"[Info] Found {len(files)} npy files in {in_dir}")
    print(f"[Info] Output dir: {out_dir}")
    print(f"[Info] Using key {use_key} from {w_npz} (shape {W.shape})")

    for fp in tqdm(files):
        name = os.path.basename(fp)
        out_fp = os.path.join(out_dir, name)

        if (not overwrite) and os.path.exists(out_fp):
            continue

        img = np.load(fp)  # expect [H,W,C]
        if img.ndim != 3:
            raise ValueError(f"{fp} expected HWC npy, got shape {img.shape}")

        # 复刻你原数据处理：抽115再裁95
        img95 = select_115_then_crop95(img)

        # 应用 W：95 -> 6
        img6 = apply_W(img95, W)  # [H,W,6]

        np.save(out_fp, img6)

    print("[OK] Done.")

if __name__ == "__main__":
    # 举例：
    # root_dir = "/data1/suliao/dataset/"  (里面有 visible/ 和 labels/)
    main(
        root_dir="/data1/suliao/",                 # TODO: 改成你的数据根目录
        w_npz="W_6x95_from_excel_T.npz",           # TODO: 你的W文件
        in_subdir="visible",
        out_subdir="visible_w6",
        use_key="W",
        overwrite=False
    )
