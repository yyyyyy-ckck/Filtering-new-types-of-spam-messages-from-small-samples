import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# ==================== 1. 加载数据 ====================
base = r"C:\Users\yj178\Desktop\ei论文"
df = pd.read_csv(f"{base}\\final_cleaned_data.csv")

X = df['final_content']
y = df['label']

# 划分训练集（模型一用 70% 历史数据训练）
X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# ==================== 2. TF‑IDF 特征提取 ====================
vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2, sublinear_tf=True)
X_train_vec = vec.fit_transform(X_train)

# ==================== 3. 加载混合测试集 ====================
test = pd.read_csv(f"{base}\\test_set_with_novel_spam.csv")
X_test = test['final_content']
y_test = test['label']
X_test_vec = vec.transform(X_test)

# ==================== 4. SVM 模型一 ====================
svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
svm.fit(X_train_vec, y_train)
pred_svm = svm.predict(X_test_vec)

print("==================== 模型一 (SVM) ====================")
print(classification_report(y_test, pred_svm, target_names=['正常短信', '垃圾短信']))
print("混淆矩阵:\n", confusion_matrix(y_test, pred_svm))

# ==================== 5. 逻辑回归模型一 ====================
lr = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
lr.fit(X_train_vec, y_train)
pred_lr = lr.predict(X_test_vec)

print("\n==================== 模型一 (逻辑回归) ====================")
print(classification_report(y_test, pred_lr, target_names=['正常短信', '垃圾短信']))
print("混淆矩阵:\n", confusion_matrix(y_test, pred_lr))