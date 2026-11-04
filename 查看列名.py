import pandas as pd

base_path = r"C:\Users\yj178\Desktop\ei论文"

# 查看 seed_20_per_class.csv 的列名和前两行
df_seed = pd.read_csv(f"{base_path}\\seed_20_per_class.csv")
print("=== seed_20_per_class.csv 列名 ===")
print(df_seed.columns.tolist())
print(df_seed.head(2))

# 查看 test_set.csv 的列名
df_test = pd.read_csv(f"{base_path}\\test_set.csv")
print("\n=== test_set.csv 列名 ===")
print(df_test.columns.tolist())
print(df_test.head(2))