import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os

def process_refractive_csv(input_path, output_path=None):
    """
    读取 RefractiveIndex.info 的分段 CSV，转换为标准格式。
    兼容性修复: 
    1. 自动处理 BOM 头 (utf-8-sig)
    2. 忽略大小写和空格
    3. 输出列: nm, n, k, wl
    """
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + "_processed.csv"

    print(f"正在处理: {input_path}")
    
    # --- 修复点 1: 使用 utf-8-sig 自动去除 BOM 头 ---
    try:
        with open(input_path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        # 如果不是 utf-8，尝试 gbk (防止中文系统生成的 csv 乱码)
        with open(input_path, 'r', encoding='gbk') as f:
            lines = f.readlines()
    
    data_n = []
    data_k = []
    current_mode = None  # 'n' 或 'k'
    
    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line: continue
        
        # 拆分并清洗数据
        parts = [p.strip().lower() for p in line.split(',')]
        
        # --- 修复点 2: 更鲁棒的表头检测 ---
        # 只要第一列包含 'wl' 或 'wavelength' 就认为是表头
        if len(parts) >= 2:
            col0 = parts[0]
            col1 = parts[1]
            
            # 检测波长列标记
            if 'wl' in col0 or 'wavelength' in col0:
                if 'n' == col1:
                    current_mode = 'n'
                    print(f"  [第 {line_idx+1} 行] -> 切换到 n 模式")
                    continue
                elif 'k' == col1:
                    current_mode = 'k'
                    print(f"  [第 {line_idx+1} 行] -> 切换到 k 模式")
                    continue
        
        # 解析数据
        try:
            wl_val = float(parts[0])
            val = float(parts[1])
            
            if current_mode == 'n':
                data_n.append([wl_val, val])
            elif current_mode == 'k':
                data_k.append([wl_val, val])
        except (ValueError, IndexError):
            continue

    # 3. 数据整合
    if not data_n:
        # 调试信息: 如果还是失败，打印前几行看看究竟长啥样
        print(f"错误: 在文件 {input_path} 中未找到 n 数据。")
        print("调试 - 文件前 3 行内容:")
        for i in range(min(3, len(lines))):
            print(f"  Line {i}: {repr(lines[i])}")
        return

    df_n = pd.DataFrame(data_n, columns=['wl', 'n'])
    
    # 处理 k 数据
    if not data_k:
        print("  - 提示: 未找到 k 数据，默认为 0 (透明材料)。")
        df_n['k'] = 0.0
    else:
        df_k = pd.DataFrame(data_k, columns=['wl', 'k'])
        # 插值对齐
        f_k = interp1d(df_k['wl'], df_k['k'], bounds_error=False, fill_value=0.0)
        df_n['k'] = f_k(df_n['wl'])
        print(f"  - 已合并 k 数据 ({len(df_k)} 行)。")

    # 4. 生成 nm 列 (假设原始 wl 单位是 um)
    df_n['nm'] = df_n['wl'] * 1000.0
    
    # 5. 调整列顺序: nm, n, k, wl
    final_df = df_n[['nm', 'n', 'k', 'wl']]
    
    # 保存
    final_df.to_csv(output_path, index=False)
    print(f"成功保存至: {output_path}")
    print("-" * 30)


if __name__ == "__main__":
    # 替换为你实际下载的文件名
    # 你可以写一个列表批量处理
    
    # 示例：假设你下载的文件叫 'PET_raw.csv'
    # process_refractive_csv('./data_new/pvc.csv', './data_new/PC.csv')
    
    # 如果你想批量处理当前文件夹下所有的 csv:
    import glob
    for file in glob.glob("*.csv"):
        if "processed" not in file: # 防止重复处理
            process_refractive_csv(file)
    #pass