import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from transformers import AutoTokenizer, AutoModel

# ================== 0. 本地 BERT 模型路径 ==================
MODEL_PATH = r"C:\Users\yj178\Desktop\ei论文\bert-base-chinese"

def get_bert_embedding(texts, batch_size=16):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH)
    model.eval()
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_emb)
    return np.concatenate(embeddings, axis=0)

# ================== 1. 准备数据 ==================
base = r"C:\Users\yj178\Desktop\ei论文"
df_all = pd.read_csv(f"{base}\\final_cleaned_data.csv")

# ================== 2. 加载新型诈骗并抽取种子 ==================
novel = pd.read_csv(f"{base}\\已清理的新型诈骗短信.csv")
if 'content' in novel.columns and 'final_content' not in novel.columns:
    novel.rename(columns={'content': 'final_content'}, inplace=True)

np.random.seed(42)
seed_novel = novel.sample(n=20, random_state=42)

# ================== 3. 构建纯 BERT 模型的训练集（与模型二完全相同） ==================
normal = df_all[df_all['label'] == 0]
old_spam = df_all[df_all['label'] == 1]

neg_normal = normal.sample(n=200, random_state=42)[['final_content']].assign(label=0)
neg_old = old_spam.sample(n=200, random_state=42)[['final_content']].assign(label=0)
pos = seed_novel[['final_content']].assign(label=1)

train_data = pd.concat([neg_normal, neg_old, pos], ignore_index=True)

# ================== 4. BERT 特征提取 + SMOTE + 训练 SVM ==================
print("正在提取训练集的 BERT 特征...")
X_train_bert = get_bert_embedding(train_data['final_content'].tolist())
y_train = train_data['label']

smote = SMOTE(sampling_strategy={1: 200}, random_state=42, k_neighbors=2)
X_train_res, y_train_res = smote.fit_resample(X_train_bert, y_train)

model_bert = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
model_bert.fit(X_train_res, y_train_res)

# ================== 5. 加载测试集并剔除种子 ==================
test = pd.read_csv(f"{base}\\test_set_with_novel_spam.csv")
test_eval = test[~test['final_content'].isin(seed_novel['final_content'])]
X_test_raw = test_eval['final_content'].tolist()
y_true = test_eval['label']  # 0=正常, 1=垃圾（含旧垃圾与新型诈骗）

# ================== 6. 对全部测试短信提取 BERT 特征并预测 ==================
print("正在提取测试集的 BERT 特征...")
X_test_bert = get_bert_embedding(X_test_raw)
pred_bert = model_bert.predict(X_test_bert)  # 0=非新型诈骗, 1=新型诈骗

# ================== 7. 评估纯 BERT 模型（无级联） ==================
# 注意：纯 BERT 模型只输出“是否新型诈骗”，直接作为最终分类结果
# 这意味着旧垃圾短信如果被模型判为0，会被错误放行；正常短信若判为1，会被错误拦截
print("\n==================== 纯 BERT 模型（无级联）—— 原始输出 ====================")
print(classification_report(y_true, pred_bert, target_names=['非新型诈骗', '新型诈骗']))
print("混淆矩阵:\n", confusion_matrix(y_true, pred_bert))

# 由于论文关心的是正常短信与垃圾短信的二元分类，我们将预测结果直接映射为最终拦截决策
print("\n==================== 纯 BERT 模型 还原为二元分类 ====================")
print(classification_report(y_true, pred_bert, target_names=['正常短信', '垃圾短信']))
print("混淆矩阵:\n", confusion_matrix(y_true, pred_bert))