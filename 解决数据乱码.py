import os
import pandas as pd

# 1. 你的文件夹路径
folder_path = r"C:\Users\yj178\Desktop\ei论文\Telecom_Fraud_Texts_5-main"

# 2. 定义智能转换函数
def fix_all_csv_encodings(directory):
    files = os.listdir(directory)
    csv_files = [f for f in files if f.endswith('.csv')]

    print(f"🔍 发现 {len(csv_files)} 个 CSV 文件，开始智能修复...\n")

    for filename in csv_files:
        file_path = os.path.join(directory, filename)
        
        # 定义读取成功的标记
        df = None
        encoding_used = ""

        # --- 尝试 1：用 UTF-8 读取 ---
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            encoding_used = "UTF-8"
        except UnicodeDecodeError:
            # --- 尝试 2：如果 UTF-8 失败，用 GBK 读取 ---
            try:
                df = pd.read_csv(file_path, encoding='gbk')
                encoding_used = "GBK"
            except Exception as e:
                print(f"❌ 彻底失败: {filename} 无法用 UTF-8 或 GBK 读取。")
                continue

        # --- 如果读取成功，统一保存为 UTF-8 ---
        if df is not None:
            # 覆盖原文件，保存为标准 UTF-8
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"✅ 已修复: {filename} (原编码: {encoding_used}) -> 已转为标准 UTF-8")

# 3. 执行修复
fix_all_csv_encodings(folder_path)

print("\n🎉 所有文件处理完毕！现在可以用 Excel 或 Python 正常打开了。")