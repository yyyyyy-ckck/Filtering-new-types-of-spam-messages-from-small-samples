import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# 其他需要的导入...

# ==================== 1. 读取数据 ====================
file_path = r"C:\Users\yj178\Desktop\ei论文\final_cleaned_data.csv"
df = pd.read_csv(file_path)

text_col = 'final_content'
label_col = 'label'

print(f"数据集总大小: {len(df)}")
print(f"类别分布:\n{df[label_col].value_counts()}")

# ==================== 2. 划分测试集 (30%) ====================
df_train_pool, df_test = train_test_split(
    df,
    test_size=0.3,
    stratify=df[label_col],
    random_state=42
)

print(f"\n训练候选池大小: {len(df_train_pool)}")
print(f"测试集大小: {len(df_test)}")

# ==================== 3. 保存测试集和训练池 ====================
save_dir = r"C:\Users\yj178\Desktop\ei论文"

df_test.to_csv(f"{save_dir}\\test_set.csv", index=False, encoding='utf-8-sig')
df_train_pool.to_csv(f"{save_dir}\\train_pool.csv", index=False, encoding='utf-8-sig')

# ==================== 4. 手动抽取每类20条（保留原始索引） ====================
np.random.seed(42)
seed_parts = []
for label_val in df_train_pool[label_col].unique():
    sub_df = df_train_pool[df_train_pool[label_col] == label_val]
    n_sample = min(20, len(sub_df))
    sampled = sub_df.sample(n=n_sample, random_state=42)
    seed_parts.append(sampled)

df_labeled_seed = pd.concat(seed_parts, ignore_index=False)  # 关键：保留原索引

print("\n种子集列名:", df_labeled_seed.columns.tolist())
print("种子集每类样本数:")
print(df_labeled_seed[label_col].value_counts())

df_labeled_seed.to_csv(f"{save_dir}\\seed_20_per_class.csv", index=False, encoding='utf-8-sig')

# ==================== 5. 未标注池 ====================
# 从训练池中删除种子集对应的行（基于索引）
df_unlabeled_pool = df_train_pool.drop(df_labeled_seed.index)
df_unlabeled_pool.to_csv(f"{save_dir}\\unlabeled_pool.csv", index=False, encoding='utf-8-sig')

print(f"\n未标注池大小: {len(df_unlabeled_pool)}")
print(f"所有文件已保存至: {save_dir}")