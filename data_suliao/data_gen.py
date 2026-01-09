import pandas as pd
import numpy as np
import random
import os
from tqdm import tqdm
import pickle as pkl
import numpy as np
from numpy import pi
import colour
import pandas as pd
import colour
import pickle as pkl
from tmm import coh_tmm, inc_tmm
from scipy.interpolate import interp1d
from colour import SDS_ILLUMINANTS, SpectralDistribution
from colour.colorimetry import MSDS_CMFS
from colour.plotting import plot_single_colour_swatch, ColourSwatch, plot_chromaticity_diagram_CIE1931
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib as mpl
import os
import itertools
from multiprocessing import Pool
import pyswarms as ps
from colour.difference import delta_E, delta_E_CIE2000



# 假设你之前的 spectrum 和 load_materials 函数在这个文件中，或者从其他文件导入
# from your_script import load_materials, spectrum, mats, wavelengths

# ==========================================
# 1. 配置参数
# ==========================================
NUM_SAMPLES = 1000       # 生成多少个样本 (根据需要修改，例如 100,000)
MAX_LAYERS = 20          # 最大层数
MIN_LAYERS = 1           # 最小层数
THICKNESS_MIN = 5        # 最小厚度 (nm)
THICKNESS_MAX = 250      # 最大厚度 (nm)
THICKNESS_STEP = 5       # 厚度步长 (nm)

# 定义所有可用材料 (与你的 mats 列表一致)
# MATERIALS_LIST = ['Al', 'Al2O3', 'AlN', 'Ge', 'HfO2', 'ITO', 'MgF2', 'MgO', 'Si', 
#                   'Si3N4', 'SiO2', 'Ta2O5', 'TiN', 'TiO2', 'ZnO', 'ZnS', 'ZnSe']
MATERIALS_LIST = ['abs', 'pc', 'pe', 'pet', 'pvc','Glass_Substrate']

# 定义厚度选项
THICKNESS_OPTIONS = list(range(THICKNESS_MIN, THICKNESS_MAX + 1, THICKNESS_STEP))

mats = ['abs', 'pc', 'pe', 'pet', 'pvc','Glass_Substrate']
thicks = [str(i) for i in range(5, 255, 5 )]

lamda_low = 0.9
lamda_high = 1.7
wavelengths = np.arange(lamda_low, lamda_high+1e-3, 0.01)


def load_materials(all_mats = mats, wavelengths = wavelengths, DATABASE = './data_new'):
    '''
    Load material nk and return corresponding interpolators.

    Return:
        nk_dict: dict, key -- material name, value: n, k in the 
        self.wavelength range
    '''
    nk_dict = {}

    for mat in all_mats:
        nk = pd.read_csv(os.path.join(DATABASE, mat + '.csv'))
        nk.dropna(inplace=True)

        wl = nk['wl'].to_numpy()
        index_n = nk['n'].to_numpy()
        index_k = nk['k'].to_numpy()

        n_fn = interp1d(
                wl, index_n,  bounds_error=False, fill_value='extrapolate', kind=3)
        k_fn = interp1d(
                wl, index_k,  bounds_error=False, fill_value='extrapolate', kind=1)
            
        nk_dict[mat] = n_fn(wavelengths) + 1j*k_fn(wavelengths)

    return nk_dict

def spectrum(materials, thickness, pol = 's', theta=0,  wavelengths = wavelengths, nk_dict = {}, substrate = 'Glass_Substrate', substrate_thick = 500000):
    '''
    Input:
        metal materials: list  
        thickness: list
        theta: degree, the incidence angle

    Return:
        All_results: dictionary contains R, T, A, RGB, LAB
    '''
    #aa = time.time()
    degree = pi/180
    theta = theta *degree
    wavess = (1e3 * wavelengths).astype('int')

        
    thickness = [np.inf] + thickness + [substrate_thick, np.inf]

    R, T, A = [], [], []
    inc_list = ['i'] + ['c']*len(materials) + ['i', 'i']
    for i, lambda_vac in enumerate(wavess):

        n_list = [1] + [nk_dict[mat][i] for mat in materials] + [nk_dict[substrate][i], 1]

        res = inc_tmm(pol, n_list, thickness, inc_list, theta, lambda_vac)

        R.append(res['R'])
        T.append(res['T'])

    # thickness = [np.inf] + thickness + [np.inf]

    # R, T, A = [], [], []
    # for i, lambda_vac in enumerate(wavess):

    #     n_list = [1] + [nk_dict[mat][i] for mat in materials] + [nk_dict[substrate][i]]

    #     res = coh_tmm(pol, n_list, thickness, theta, lambda_vac)

    #     R.append(res['R'])
    #     T.append(res['T'])

    return R + T

# ==========================================
# 2. 数据生成核心函数
# ==========================================
def generate_dataset(output_csv='optogpt_dataset.csv', output_pkl='optogpt_dataset.pkl'):
    print("--- 1. Loading Materials Data (nk) ---")
    try:
        # 加载材料折射率数据 (确保 ./nk 文件夹里有对应的 csv)
        nk_dict = load_materials() 
        print("Materials loaded successfully.")
    except Exception as e:
        print(f"Error loading materials: {e}")
        return

    data_rows = []
    
    print(f"--- 2. Generating {NUM_SAMPLES} Random Structures ---")
    
    for i in tqdm(range(NUM_SAMPLES)):
        # A. 随机生成结构
        num_layers = random.randint(MIN_LAYERS, MAX_LAYERS)
        
        current_materials = []
        current_thicknesses = []
        structure_tokens = [] # 用于存储 "Material_Thickness" 格式的字符串
        
        for _ in range(num_layers):
            # 随机选择材料和厚度
            mat = random.choice(MATERIALS_LIST)
            thick = random.choice(THICKNESS_OPTIONS)
            
            current_materials.append(mat)
            current_thicknesses.append(thick)
            structure_tokens.append(f"{mat}_{thick}")
        
        # B. 计算光谱 (R 和 T)
        # 注意：spectrum 函数返回的是 [R_list] + [T_list]
        try:
            # 调用你的 spectrum 函数
            # 这里的 thickness 需要注意单位，tmm 库通常根据波长单位计算
            # 你的 wavelengths 是 0.4-1.1 (um)，spectrum 内部转成了 nm (wavess)
            # 所以 thickness 传入 nm 数值是匹配的
            spectra = spectrum(current_materials, current_thicknesses, nk_dict=nk_dict)
            
            # 检查返回数据长度是否正确 (71个波长点 * 2 = 142)
            # 0.4 - 1.1 um, step 0.01 -> 71 points
            expected_points = len(wavelengths)
            if len(spectra) != expected_points * 2:
                print(f"Skipping sample {i}: Spectrum length mismatch.")
                continue

            # C. 构建数据行
            row = {
                'structure': structure_tokens,  # 存储列表
                'lengths': num_layers,          # 结构长度
                'num_layers': num_layers,       # 层数
            }
            
            # 将 R 和 T 的列表展开为列
            # 波长范围: 400nm 到 1100nm，步长 10nm
            wl_nm = (wavelengths * 1000).astype(int)
            
            # 前半部分是 R
            for idx, r_val in enumerate(spectra[:expected_points]):
                row[f'R_{wl_nm[idx]}nm'] = r_val
                
            # 后半部分是 T
            for idx, t_val in enumerate(spectra[expected_points:]):
                row[f'T_{wl_nm[idx]}nm'] = t_val
            
            data_rows.append(row)

        except Exception as e:
            # 捕获计算中的错误 (例如矩阵奇异值等)
            # print(f"Error in sample {i}: {e}")
            continue

    # ==========================================
    # 3. 保存数据
    # ==========================================
    print("--- 3. Saving Data ---")
    df = pd.DataFrame(data_rows)
    
    # 保存 CSV (方便查看)
    df.to_csv(output_csv, index=True) # index=True 会生成 Unnamed: 0 列
    
    # 保存 Pickle (保留 List 对象结构，方便后续 PrepareData 读取)
    # 注意：OptoGPT 的 PrepareData 类似乎分别读取 structure 和 spectrum 的 pkl
    # 这里我们生成一个综合的 DataFrame pkl
    df.to_pickle(output_pkl)
    
    # 如果你需要为了 OptoGPT 的 PrepareData 类生成分离的 pkl:
    structures_only = df['structure'].tolist()
    spectra_only = df.drop(columns=['structure', 'lengths', 'num_layers', 'Unnamed: 0'], errors='ignore').values.tolist()
    
    with open('train_structure.pkl', 'wb') as f:
        pkl.dump(structures_only, f)
    with open('train_spectrum.pkl', 'wb') as f:
        pkl.dump(spectra_only, f)

    print(f"Dataset generated with {len(df)} samples.")
    print("Files saved: optogpt_dataset.csv, train_structure.pkl, train_spectrum.pkl")
    return df

# ==========================================
# 运行入口
# ==========================================
if __name__ == "__main__":
    # 确保当前目录下有 nk 文件夹和之前的 csv 文件
    if not os.path.exists('./data_new'):
        os.makedirs('./data_new')
        print("Warning: './nk' directory created. Please put material CSV files inside.")
    
    # 运行生成
    df_result = generate_dataset()
    if df_result is not None:
        print(df_result.head())