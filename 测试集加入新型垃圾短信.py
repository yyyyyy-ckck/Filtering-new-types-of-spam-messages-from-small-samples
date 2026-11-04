import pandas as pd

base = r"C:\Users\yj178\Desktop\ei论文"

# 读取原测试集
test = pd.read_csv(f"{base}\\test_set.csv")
# 读取新型诈骗短信
novel = pd.read_csv(f"{base}\\已清理的新型诈骗短信.csv")

print(f"原测试集: {len(test)} 条")
print(f"新型短信: {len(novel)} 条")

# 合并
combined = pd.concat([test, novel], ignore_index=True)
print(f"合并后: {len(combined)} 条")

# 保存新测试集（不覆盖原文件）
combined.to_csv(f"{base}\\test_set_with_novel_spam.csv", index=False, encoding='utf-8-sig')
print("已保存 test_set_with_novel_spam.csv")