import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer

# ================== 0. 本地模型路径（三个文件需提前下载到这里） ==================
MODEL_PATH = r"C:\Users\yj178\Desktop\ei论文\bert-base-chinese"
# 请确保此文件夹内包含：config.json、pytorch_model.bin、vocab.txt

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

# ================== 1. 准备原始数据 ==================
base = r"C:\Users\yj178\Desktop\ei论文"
df_all = pd.read_csv(f"{base}\\final_cleaned_data.csv")

# ================== 2. 训练模型一（充分样本基线，仍用 TF-IDF） ==================
X_m1 = df_all['final_content']
y_m1 = df_all['label']
X_train_m1, _, y_train_m1, _ = train_test_split(
    X_m1, y_m1, test_size=0.3, stratify=y_m1, random_state=42
)

vec_m1 = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2, sublinear_tf=True)
X_train_m1_vec = vec_m1.fit_transform(X_train_m1)
model1 = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
model1.fit(X_train_m1_vec, y_train_m1)

# ================== 3. 加载新型诈骗并抽取种子 ==================抽取20个新型垃圾短信作为负样本
novel = pd.read_csv(f"{base}\\已清理的新型诈骗短信.csv")
if 'content' in novel.columns and 'final_content' not in novel.columns:
    novel.rename(columns={'content': 'final_content'}, inplace=True)

np.random.seed(42)
seed_novel = novel.sample(n=20, random_state=42)

# ================== 4. 构建模型二训练集 ==================抽取正样本和负样本
normal = df_all[df_all['label'] == 0]
old_spam = df_all[df_all['label'] == 1]

neg_normal = normal.sample(n=200, random_state=42)[['final_content']].assign(label=0)
neg_old = old_spam.sample(n=200, random_state=42)[['final_content']].assign(label=0)
pos = seed_novel[['final_content']].assign(label=1)

train_m2 = pd.concat([neg_normal, neg_old, pos], ignore_index=True)

# ================== 5. AI 特征工程：用 BERT 向量替代 TF-IDF ==================
print("正在提取 BERT 特征（从本地加载模型）...")
X_train_m2_bert = get_bert_embedding(train_m2['final_content'].tolist())
y_train_m2 = train_m2['label']

# SMOTE 过采样
smote = SMOTE(sampling_strategy={1: 200}, random_state=42, k_neighbors=2)
X_train_m2_res, y_train_m2_res = smote.fit_resample(X_train_m2_bert, y_train_m2)

# 训练模型二（SVM + BERT 特征）
model2 = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
model2.fit(X_train_m2_res, y_train_m2_res)

# ================== 6. 加载测试集并剔除种子 ==================去除20条新型垃圾短信
test = pd.read_csv(f"{base}\\test_set_with_novel_spam.csv")
test_eval = test[~test['final_content'].isin(seed_novel['final_content'])]
X_test_raw = test_eval['final_content'].tolist()
y_test = test_eval['label']

# ================== 7. 级联预测 ==================
# 模型一预测
X_test_m1_vec = vec_m1.transform(X_test_raw)
pred_m1 = model1.predict(X_test_m1_vec)

# 对模型一判为0的样本，用模型二筛查（提取 BERT 特征）
mask_normal = pred_m1 == 0
if mask_normal.sum() > 0:
    X_test_m2_raw = [X_test_raw[i] for i in range(len(X_test_raw)) if mask_normal[i]]
    print(f"正在为 {len(X_test_m2_raw)} 条放行短信提取 BERT 特征...")
    X_test_m2_bert = get_bert_embedding(X_test_m2_raw)
    pred_m2_on_normal = model2.predict(X_test_m2_bert)
else:
    pred_m2_on_normal = np.array([], dtype=int)

pred_final = pred_m1.copy()
pred_final[mask_normal] = pred_m2_on_normal

# ================== 8. 评估对比 ==================
y_true = test_eval['label']

print("\n==================== 仅模型一 ====================")
print(classification_report(y_true, pred_m1, target_names=['正常短信', '垃圾短信']))
print("混淆矩阵:\n", confusion_matrix(y_true, pred_m1))

print("\n==================== 级联模型（模型一 TF-IDF + 模型二 BERT） ====================")
print(classification_report(y_true, pred_final, target_names=['正常短信', '垃圾短信']))
print("混淆矩阵:\n", confusion_matrix(y_true, pred_final))