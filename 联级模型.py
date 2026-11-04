import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

base = r"C:\Users\yj178\Desktop\ei论文"

# ================== 1. 准备原始数据 ==================
df_all = pd.read_csv(f"{base}\\final_cleaned_data.csv")

# ================== 2. 训练模型一（充分样本基线） ==================
X_m1 = df_all['final_content']
y_m1 = df_all['label']
X_train_m1, _, y_train_m1, _ = train_test_split(
    X_m1, y_m1, test_size=0.3, stratify=y_m1, random_state=42
)

vec_m1 = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2, sublinear_tf=True)
X_train_m1_vec = vec_m1.fit_transform(X_train_m1)
model1 = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
model1.fit(X_train_m1_vec, y_train_m1)

# ================== 3. 加载新型诈骗并抽取种子 ==================
novel = pd.read_csv(f"{base}\\已清理的新型诈骗短信.csv")
if 'content' in novel.columns and 'final_content' not in novel.columns:
    novel.rename(columns={'content': 'final_content'}, inplace=True)

# 随机抽取20条作为训练正样本
np.random.seed(42)
seed_novel = novel.sample(n=20, random_state=42)
# 剩余480条作为测试正样本（但不需单独保存，因为测试集已包含它们，只要剔除种子即可）

# ================== 4. 构建模型二训练集 ==================
normal = df_all[df_all['label'] == 0]
old_spam = df_all[df_all['label'] == 1]

# 负样本：200正常 + 200旧垃圾
neg_normal = normal.sample(n=200, random_state=42)[['final_content']].assign(label=0)
neg_old = old_spam.sample(n=200, random_state=42)[['final_content']].assign(label=0)
# 正样本：种子20条，label=1
pos = seed_novel[['final_content']].assign(label=1)

train_m2 = pd.concat([neg_normal, neg_old, pos], ignore_index=True)

# SMOTE 过采样正类到200
vec_m2 = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2, sublinear_tf=True)
X_train_m2_vec = vec_m2.fit_transform(train_m2['final_content'])
y_train_m2 = train_m2['label']

smote = SMOTE(sampling_strategy={1: 200}, random_state=42, k_neighbors=2)
X_train_m2_res, y_train_m2_res = smote.fit_resample(X_train_m2_vec, y_train_m2)

model2 = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
model2.fit(X_train_m2_res, y_train_m2_res)

# ================== 5. 加载测试集并剔除种子 ==================
test = pd.read_csv(f"{base}\\test_set_with_novel_spam.csv")
# 剔除用于训练模型二的20条种子
test_eval = test[~test['final_content'].isin(seed_novel['final_content'])]
X_test = test_eval['final_content']
y_test = test_eval['label']

print(f"测试集大小（剔除种子后）: {len(test_eval)}")
print(f"测试集类别分布:\n{y_test.value_counts()}")

# ================== 6. 级联预测 ==================
# 模型一预测
X_test_m1_vec = vec_m1.transform(X_test)
pred_m1 = model1.predict(X_test_m1_vec)

# 对模型一判为0的样本，用模型二筛查
mask_normal = pred_m1 == 0
if mask_normal.sum() > 0:
    X_test_m2_vec = vec_m2.transform(X_test[mask_normal])
    pred_m2_on_normal = model2.predict(X_test_m2_vec)
else:
    pred_m2_on_normal = np.array([], dtype=int)

pred_final = pred_m1.copy()
pred_final[mask_normal] = pred_m2_on_normal

# ================== 7. 评估对比 ==================
print("\n==================== 仅模型一 ====================")
print(classification_report(y_test, pred_m1, target_names=['正常/旧垃圾', '新型诈骗']))
print("混淆矩阵:\n", confusion_matrix(y_test, pred_m1))

print("\n==================== 级联模型（模型一+模型二） ====================")
print(classification_report(y_test, pred_final, target_names=['正常/旧垃圾', '新型诈骗']))
print("混淆矩阵:\n", confusion_matrix(y_test, pred_final))
# ==================== 5. 对比评估 ====================
y_true = test_eval['label']   # 确保 y_true 已定义

print("==================== 仅模型一 ====================")
print(classification_report(y_true, pred_m1, target_names=['正常短信', '垃圾短信']))
print("混淆矩阵:\n", confusion_matrix(y_true, pred_m1))

print("\n==================== 级联模型（模型一+模型二）还原为二元分类 ====================")
print(classification_report(y_true, pred_final, target_names=['正常短信', '垃圾短信']))
print("混淆矩阵:\n", confusion_matrix(y_true, pred_final))