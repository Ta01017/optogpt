import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os

def extract_pe_abs_data(input_csv, start_nm=900, end_nm=1700):
    print(f"--- 正在提取 PE 和 ABS 数据 (范围: {start_nm}-{end_nm} nm) ---")
    
    # 1. 准备目标波长网格 (nm -> um)
    target_nm = np.arange(start_nm, end_nm + 1, 1)
    target_um = target_nm / 1000.0
    
    # 2. 读取原始 CSV
    try:
        # header=0 读取第一行作为表头
        df_raw = pd.read_csv(input_csv, header=0)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return

    # 3. 自动定位 PE 和 ABS 的列索引
    # CSV结构假定是分块的: [Name, wl, n, k, empty, Name...]
    cols = df_raw.columns.tolist()
    mat_indices = {}
    
    # 只需要 PE 和 ABS
    target_materials = ['PE', 'ABS']
    
    for i, col in enumerate(cols):
        clean_name = str(col).strip()
        if clean_name in target_materials:
            mat_indices[clean_name] = i
            
    if not mat_indices:
        print("❌ 未在文件中找到 PE 或 ABS。请检查 CSV 格式。")
        return

    print(f"已定位材料索引: {mat_indices}")

    # 4. 循环处理每个材料
    for mat_name, idx in mat_indices.items():
        try:
            # 提取三列: [wavelength, n, k]
            # 对应的索引是 idx+1, idx+2, idx+3
            sub_df = df_raw.iloc[:, idx+1:idx+4].copy()
            sub_df.columns = ['wl', 'n', 'k']
            
            # 数据清洗: 转为数值型，去除无效行
            sub_df = sub_df.apply(pd.to_numeric, errors='coerce').dropna()
            
            if sub_df.empty:
                print(f"⚠️ {mat_name} 数据为空，跳过。")
                continue

            # ==========================================
            # 核心插值策略 (确认: n=Cubic, k=Linear)
            # ==========================================
            
            # 1. 折射率 n: 使用三次样条 (Cubic)
            # 理由: 物理色散曲线是平滑的，三次插值精度最高
            f_n = interp1d(sub_df['wl'], sub_df['n'], kind='cubic', 
                          bounds_error=False, fill_value="extrapolate")
            
            # 2. 消光系数 k: 使用线性插值 (Linear)
            # 理由: 防止在 k=0 附近因震荡产生负值
            f_k = interp1d(sub_df['wl'], sub_df['k'], kind='linear', 
                          bounds_error=False, fill_value=0.0)
            
            # 生成新数据
            n_new = f_n(target_um)
            k_new = f_k(target_um)
            
            # 物理约束修正: k 不能小于 0
            k_new[k_new < 0] = 0.0
            
            # ==========================================
            # 保存文件
            # ==========================================
            output_df = pd.DataFrame({
                'nm': target_nm,
                'n': n_new,
                'k': k_new,
                'wl': target_um
            })
            
            filename = f"{mat_name.lower()}.csv"
            output_df.to_csv(filename, index=False)
            print(f"✅ 成功生成: {filename}")
            
        except Exception as e:
            print(f"❌ 处理 {mat_name} 时出错: {e}")

    print("\n处理完成。")

# 运行入口
if __name__ == "__main__":
    # 请确保文件名与你下载的补充材料文件名一致
    csv_filename = './data_new/nd.csv'
    
    if os.path.exists(csv_filename):
        extract_pe_abs_data(csv_filename)
    else:
        print(f"错误: 找不到文件 {csv_filename}")