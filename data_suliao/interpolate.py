import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
import os

def generate_high_precision_csv(input_path, output_path=None, start_nm=900, end_nm=1700, step_nm=1):
    """
    读取 RefractiveIndex.info 的数据，使用高精度算法(Cubic Spline)重采样。
    
    参数:
        input_path: 输入 CSV 路径
        output_path: 输出 CSV 路径
        start_nm, end_nm, step_nm: 目标波长范围和步长(nm)
    """
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + "_new.csv"

    print(f"--- 正在进行高精度处理: {input_path} ---")
    
    # ==========================================
    # 1. 鲁棒读取 (处理 BOM 和分段结构)
    # ==========================================
    try:
        with open(input_path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(input_path, 'r', encoding='gbk') as f:
            lines = f.readlines()

    data_n = []
    data_k = []
    current_mode = None 

    for line in lines:
        line = line.strip()
        if not line: continue
        parts = [p.strip().lower() for p in line.split(',')]

        # 智能表头检测
        if len(parts) >= 2 and ('wl' in parts[0] or 'wavelength' in parts[0]):
            if parts[1] == 'n':
                current_mode = 'n'
                continue
            elif parts[1] == 'k':
                current_mode = 'k'
                continue
        
        try:
            wl_val = float(parts[0]) 
            val = float(parts[1])
            if current_mode == 'n':
                data_n.append([wl_val, val])
            elif current_mode == 'k':
                data_k.append([wl_val, val])
        except:
            continue
            
    # ==========================================
    # 2. 数据预处理 (去重、排序) - 高精度插值的关键
    # ==========================================
    def clean_df(data, col_name):
        if not data:
            return pd.DataFrame(columns=['wl_um', col_name])
        df = pd.DataFrame(data, columns=['wl_um', col_name])
        # 按波长排序
        df = df.sort_values(by='wl_um')
        # 去除重复波长 (取平均值)
        df = df.groupby('wl_um', as_index=False).mean()
        return df

    df_n = clean_df(data_n, 'n')
    df_k = clean_df(data_k, 'k')

    if df_n.empty:
        print(f"错误: {input_path} 中没有有效数据。")
        return

    # ==========================================
    # 3. 建立高精度插值函数
    # ==========================================
    # 目标网格 (转换为微米)
    target_nm = np.arange(start_nm, end_nm + step_nm, step_nm)
    target_wl_um = target_nm / 1000.0
    
    # --- A. 折射率 n: 使用三次样条 (Cubic) ---
    # 这种插值方法能保证一阶和二阶导数连续，曲线最平滑
    try:
        f_n = interp1d(df_n['wl_um'], df_n['n'], kind='cubic', 
                       bounds_error=False, fill_value="extrapolate")
        interp_n = f_n(target_wl_um)
    except Exception as e:
        print(f"警告: 数据点太少无法进行三次插值，降级为线性插值。({e})")
        f_n = interp1d(df_n['wl_um'], df_n['n'], kind='linear', 
                       bounds_error=False, fill_value="extrapolate")
        interp_n = f_n(target_wl_um)

    # --- B. 消光系数 k: 使用 PCHIP 或 线性 ---
    # 这里我们使用 Linear 以确保绝对的物理安全性 (防止震荡出负数)
    if not df_k.empty:
        f_k = interp1d(df_k['wl_um'], df_k['k'], kind='linear', 
                       bounds_error=False, fill_value=0.0)
        interp_k = f_k(target_wl_um)
        
        # 强制修正：物理上 k 不能小于 0
        interp_k[interp_k < 0] = 0.0
    else:
        interp_k = np.zeros_like(target_wl_um)

    # ==========================================
    # 4. 生成结果
    # ==========================================
    result_df = pd.DataFrame({
        'nm': target_nm,
        'n': interp_n,
        'k': interp_k,
        'wl': target_wl_um
    })

    # 设置高精度显示格式 (保留6位小数，k用科学计数法)
    # 注意：保存为csv时 pandas 会默认使用最高精度，这里的格式化主要用于预览
    pd.options.display.float_format = '{:.6f}'.format
    
    result_df.to_csv(output_path, index=False)
    print(f"✅ 成功! 高精度插值文件已生成: {output_path}")
    print(f"   算法: n=Cubic Spline, k=Linear")
    print(f"   范围: {start_nm}-{end_nm} nm, 步长: {step_nm} nm")
    print(result_df.head())

# ==========================================
# 运行
# ==========================================
if __name__ == "__main__":
    # 你可以在这里批量处理所有文件
    files_to_process = ['./data_new/pc.csv','./data_new/pet.csv','./data_new/pvc.csv'] # 把 pet.csv, pvc.csv 加进去
    
    for f in files_to_process:
        if os.path.exists(f):
            generate_high_precision_csv(f)
        else:
            print(f"文件不存在: {f}")