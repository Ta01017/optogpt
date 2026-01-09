import pickle as pkl
import numpy as np
import os

# ===== 路径 =====
DATA_DIR = './output'   # 你现在的数据输出目录
OUT_DIR  = './dataset'  # 训练脚本用的目录
os.makedirs(OUT_DIR, exist_ok=True)

STRUCT_FILE = os.path.join(DATA_DIR, 'total_structure.pkl')
SPEC_FILE   = os.path.join(DATA_DIR, 'total_spectrum.pkl')

# ===== 读取 =====
with open(STRUCT_FILE, 'rb') as f:
    structures = pkl.load(f)

with open(SPEC_FILE, 'rb') as f:
    spectra = pkl.load(f)

assert len(structures) == len(spectra)
N = len(structures)
print(f"Total samples: {N}")

# ===== 打乱（但保持对应）=====
indices = np.random.permutation(N)
structures = [structures[i] for i in indices]
spectra    = [spectra[i]    for i in indices]

# ===== 按比例切 =====
ratio = 0.8
split = int(N * ratio)

train_struct = structures[:split]
train_spec   = spectra[:split]

dev_struct = structures[split:]
dev_spec   = spectra[split:]

print(f"Train: {len(train_struct)} | Dev: {len(dev_struct)}")

# ===== 保存 =====
with open(os.path.join(OUT_DIR, 'Structure_train.pkl'), 'wb') as f:
    pkl.dump(train_struct, f)

with open(os.path.join(OUT_DIR, 'Spectrum_train.pkl'), 'wb') as f:
    pkl.dump(train_spec, f)

with open(os.path.join(OUT_DIR, 'Structure_dev.pkl'), 'wb') as f:
    pkl.dump(dev_struct, f)

with open(os.path.join(OUT_DIR, 'Spectrum_dev.pkl'), 'wb') as f:
    pkl.dump(dev_spec, f)

print("✔ Dataset split finished.")
