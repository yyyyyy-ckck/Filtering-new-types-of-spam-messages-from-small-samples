import pandas as pd
import jieba
import re

base = r"C:\Users\yj178\Desktop\ei论文"

# 读取原始 CSV
df = pd.read_csv(f"{base}\\new_spam_500.csv", encoding='utf-8-sig')

# 统一列名
if 'content' in df.columns and 'final_content' not in df.columns:
    df.rename(columns={'content': 'final_content'}, inplace=True)

# 文本清洗与分词函数
def clean_and_segment(text):
    text = str(text)
    # 去除 URL
    text = re.sub(r'http\S+', '', text)
    # 去除邮箱
    text = re.sub(r'\S+@\S+', '', text)
    # 去除多余空格和不可见控制字符
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', text)
    # 将连续空格合并为一个
    text = re.sub(r'\s+', ' ', text).strip()
    # 若文本为空则返回空字符串
    if not text:
        return ''
    # 使用 jieba 进行精确模式分词，结果用空格隔开
    words = jieba.lcut(text)
    return ' '.join(words)

# 应用分词
df['final_content'] = df['final_content'].apply(clean_and_segment)

# 删除分词后为空的无效行
df = df[df['final_content'].str.strip() != '']

# 确保 label 列为整数 1
df['label'] = 1

# 保存清洗后的文件
output_path = f"{base}\\new_spam_500_clean.csv"
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"清洗完成！共处理 {len(df)} 条短信，已保存至 {output_path}")