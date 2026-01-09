import pandas as pd
import numpy as np
import glob
import os
import json

# ================= 配置区域 =================
INPUT_FOLDER = r'.'  # 你的CSV文件夹路径
WAVE_MIN = 900   # 波段下限 (nm)
WAVE_MAX = 1700  # 波段上限 (nm)

# 输出文件名
OUTPUT_REPORT = 'Material_Report.csv'     # 给人类看的表格
OUTPUT_JSON = 'materials_db.json'         # 给程序读的JSON数据
OUTPUT_PY = 'materials_list_generated.py' # 生成的Python代码文件

# ================= 分类核心逻辑 =================
def get_category_label(avg_n, avg_k):
    """
    基于 900-1700nm 的物理特性进行分类
    """
    # 1. 金属或高吸收体
    if avg_k > 1.0: 
        return "METAL"
    if avg_k > 0.05: 
        return "ABSORBER_LOSSY"
    
    # 2. 透明介质 (按折射率排序)
    if avg_n < 1.6:
        return "LOW_INDEX"      # < 1.6
    elif 1.6 <= avg_n < 2.0:
        return "MEDIUM_INDEX"   # 1.6 - 2.0
    elif 2.0 <= avg_n < 3.0:
        return "HIGH_INDEX"     # 2.0 - 3.0
    elif avg_n >= 3.0:
        return "ULTRA_HIGH_INDEX" # > 3.0
    
    return "UNKNOWN"

# ================= 主处理函数 =================
def process_materials():
    # 1. 获取所有csv文件
    if not os.path.exists(INPUT_FOLDER):
        print(f"错误: 找不到文件夹 '{INPUT_FOLDER}'")
        return

    files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
    data_list = []
    
    print(f"正在扫描 {len(files)} 个材料文件...\n")

    for file_path in files:
        filename = os.path.basename(file_path)
        material_name = os.path.splitext(filename)[0]
        
        try:
            # 读取CSV (兼容无表头或不同列名)
            try:
                df = pd.read_csv(file_path)
            except:
                continue # 跳过无法读取的文件

            # 简单的列名清洗
            df.columns = [str(c).strip().lower() for c in df.columns]
            
            # 智能匹配列 (Wavelength, n, k)
            cols = df.columns
            col_wl = next((c for c in cols if any(x in c for x in ['wl', 'wave', 'lam', 'nm', 'um'])), cols[0])
            col_n = next((c for c in cols if c == 'n'), cols[1] if len(cols)>1 else None)
            col_k = next((c for c in cols if c == 'k'), cols[2] if len(cols)>2 else None)

            # 提取并转为数值
            wl = pd.to_numeric(df[col_wl], errors='coerce')
            n_vals = pd.to_numeric(df[col_n], errors='coerce')
            k_vals = pd.to_numeric(df[col_k], errors='coerce').fillna(0)

            # 单位修正: 如果波长都很小(<100)，说明是um，转为nm
            if wl.mean() < 100:
                wl = wl * 1000
            
            # 截取 900-1700nm 波段
            mask = (wl >= WAVE_MIN) & (wl <= WAVE_MAX)
            roi_n = n_vals[mask]
            roi_k = k_vals[mask]

            if len(roi_n) == 0:
                print(f"[跳过] {material_name}: 数据未覆盖 900-1700nm")
                continue

            # 计算平均值
            avg_n = float(roi_n.mean())
            avg_k = float(roi_k.mean())
            category = get_category_label(avg_n, avg_k)

            # 存入列表
            data_list.append({
                "name": material_name,
                "category": category,
                "avg_n": round(avg_n, 4),
                "avg_k": round(avg_k, 5),
                "file_path": file_path.replace("\\", "/") # 统一路径分隔符
            })
            
            print(f"处理完成: {material_name:<10} | n={avg_n:.2f} k={avg_k:.2f} -> {category}")

        except Exception as e:
            print(f"[错误] 处理 {material_name} 失败: {e}")

    # ================= 结果生成 =================
    
    # 1. 排序: 先按类别，再按折射率
    # 定义类别排序优先级
    cat_order = {"LOW_INDEX": 1, "MEDIUM_INDEX": 2, "HIGH_INDEX": 3, "ULTRA_HIGH_INDEX": 4, "ABSORBER_LOSSY": 5, "METAL": 6, "UNKNOWN": 99}
    data_list.sort(key=lambda x: (cat_order.get(x['category'], 99), x['avg_n']))

    # 2. 保存 CSV 报表 (给人看)
    df_res = pd.DataFrame(data_list)
    df_res.to_csv(OUTPUT_REPORT, index=False)
    print(f"\n[成功] CSV 报表已保存: {OUTPUT_REPORT}")

    # 3. 保存 JSON 文件 (给程序读)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)
    print(f"[成功] JSON 数据已保存: {OUTPUT_JSON}")

    # 4. 生成 Python 代码文件 (直接 import 用)
    with open(OUTPUT_PY, 'w', encoding='utf-8') as f:
        f.write("# Auto-generated material list\n")
        f.write(f"# Wavelength Range: {WAVE_MIN}-{WAVE_MAX} nm\n\n")
        f.write("ALL_MATERIALS = [\n")
        for item in data_list:
            f.write(f"    {{'name': '{item['name']}', 'category': '{item['category']}', 'n': {item['avg_n']}, 'k': {item['avg_k']}, 'path': '{item['file_path']}'}},\n")
        f.write("]\n\n")
        
        # 顺便生成一个分类字典
        f.write("# Helper dictionary by category\n")
        f.write("MATERIALS_BY_CATEGORY = {\n")
        for cat in cat_order.keys():
            mats = [m['name'] for m in data_list if m['category'] == cat]
            if mats:
                f.write(f"    '{cat}': {mats},\n")
        f.write("}\n")
        
    print(f"[成功] Python 列表代码已生成: {OUTPUT_PY}")

if __name__ == "__main__":
    process_materials()