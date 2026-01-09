import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os
import glob

def generate_high_precision_csv(input_path, output_path=None, start_nm=900, end_nm=1700, step_nm=1):
    """
    通用高精度插值脚本
    支持：
    1. 已经清洗过的标准 CSV (含 columns: nm/wl, n, k)
    2. RefractiveIndex.info 的原始分段 CSV
    """
    if output_path is None:
        # 避免覆盖原文件，建议输出到新文件夹或改名
        if not os.path.exists('./processed'):
            os.makedirs('./processed')
        filename = os.path.basename(input_path)
        output_path = os.path.join('./processed', filename)

    print(f"--- 正在处理: {input_path} ---")
    
    df_n = pd.DataFrame()
    df_k = pd.DataFrame()
    
    # ==========================================
    # 策略 A: 尝试直接读取为标准 DataFrame
    # ==========================================
    try:
        # 尝试读取
        df_raw = pd.read_csv(input_path)
        cols = [c.lower() for c in df_raw.columns]
        df_raw.columns = cols
        
        # 检查是否包含关键列
        has_n = 'n' in cols
        has_k = 'k' in cols
        # 寻找波长列 (nm, wl, wavelength, etc.)
        wl_col = next((c for c in cols if c in ['nm', 'wl', 'wavelength', 'wl_um']), None)
        
        if wl_col and has_n:
            print(" -> 识别为: 标准 CSV 格式")
            # 统一单位为 um
            if 'nm' in wl_col: # 假设是 nm
                df_raw['wl_um'] = df_raw[wl_col] / 1000.0
            else: # 假设是 um
                df_raw['wl_um'] = df_raw[wl_col]
            
            df_n = df_raw[['wl_um', 'n']].dropna()
            if has_k:
                df_k = df_raw[['wl_um', 'k']].dropna()
        else:
            raise ValueError("不是标准格式")
            
    except Exception:
        # ==========================================
        # 策略 B: 回退到原始解析逻辑 (RefractiveIndex.info)
        # ==========================================
        print(" -> 识别为: 原始/复杂 CSV 格式 (尝试解析文本)")
        try:
            with open(input_path, 'r', encoding='utf-8-sig') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(input_path, 'r', encoding='gbk') as f:
                lines = f.readlines()

        data_n_list = []
        data_k_list = []
        current_mode = None 

        for line in lines:
            line = line.strip()
            if not line: continue
            parts = [p.strip().lower() for p in line.split(',')]
            if not parts: continue

            # 检测表头切换模式
            if len(parts) >= 2:
                # 检查第一列是否是波长
                if any(x in parts[0] for x in ['wl', 'wave', 'nm']):
                    if 'n' in parts[1]:
                        current_mode = 'n'
                        continue
                    elif 'k' in parts[1]:
                        current_mode = 'k'
                        continue
            
            # 读取数据
            try:
                # 假设第一列是波长，第二列是数值
                val_x = float(parts[0])
                val_y = float(parts[1])
                
                if current_mode == 'n':
                    data_n_list.append([val_x, val_y])
                elif current_mode == 'k':
                    data_k_list.append([val_x, val_y])
            except:
                continue
        
        if data_n_list:
            df_n = pd.DataFrame(data_n_list, columns=['wl_um', 'n'])
        if data_k_list:
            df_k = pd.DataFrame(data_k_list, columns=['wl_um', 'k'])

    # ==========================================
    # 检查数据有效性
    # ==========================================
    if df_n.empty:
        print(f"❌ 错误: {input_path} 中未提取到有效数据 (n)。跳过。")
        return

    # ==========================================
    # 高精度插值 (Core)
    # ==========================================
    # 目标网格
    target_nm = np.arange(start_nm, end_nm + step_nm, step_nm)
    target_wl_um = target_nm / 1000.0
    
    # 1. 插值 n (Cubic Spline)
    try:
        # 按波长排序去重
        df_n = df_n.sort_values('wl_um').groupby('wl_um', as_index=False).mean()
        
        if len(df_n) > 3:
            f_n = interp1d(df_n['wl_um'], df_n['n'], kind='cubic', 
                          bounds_error=False, fill_value="extrapolate")
        else:
            f_n = interp1d(df_n['wl_um'], df_n['n'], kind='linear', 
                          bounds_error=False, fill_value="extrapolate")
        interp_n = f_n(target_wl_um)
    except Exception as e:
        print(f"⚠️ 插值 n 出错: {e}")
        return

    # 2. 插值 k (Linear)
    if not df_k.empty:
        df_k = df_k.sort_values('wl_um').groupby('wl_um', as_index=False).mean()
        f_k = interp1d(df_k['wl_um'], df_k['k'], kind='linear', 
                      bounds_error=False, fill_value=0.0)
        interp_k = f_k(target_wl_um)
        interp_k[interp_k < 0] = 0 # 修正负值
    else:
        interp_k = np.zeros_like(target_wl_um)

    # ==========================================
    # 保存结果
    # ==========================================
    result_df = pd.DataFrame({
        'nm': target_nm,
        'n': interp_n,
        'k': interp_k,
        'wl': target_wl_um
    })
    
    result_df.to_csv(output_path, index=False)
    print(f"✅ 已生成: {output_path} (Range: {start_nm}-{end_nm} nm)")

if __name__ == "__main__":
    # 指定你的文件夹路径
    source_folder = "."  # 这里改成你放 csv 的文件夹
    csv_files = glob.glob(os.path.join(source_folder, "*.csv"))
    
    if not csv_files:
        print(f"在 {source_folder} 中没找到 CSV 文件。")
    
    for f in csv_files:
        # 跳过已经生成的 Glass_Substrate 避免重复处理
        if "Glass" in f: 
            continue 
        generate_high_precision_csv(f)