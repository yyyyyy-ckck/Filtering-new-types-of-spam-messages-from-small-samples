import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ==================== 1. 加载原始全量数据（仅用于训练） ====================
file_path = r"C:\Users\yj178\Desktop\ei论文\final_cleaned_data.csv"
df = pd.read_csv(file_path)

X_raw = df['final_content']
y = df['label']

# 划分出训练集（70%），丢弃产生的测试集（30%）
X_train_raw, _, y_train, _ = train_test_split(
    X_raw, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

print(f"原始数据集总大小: {len(df)}")
print(f"训练集大小: {len(X_train_raw)}")

# ==================== 2. 加载新的测试集（包含新型诈骗短信） ====================
test_path = r"C:\Users\yj178\Desktop\ei论文\test_set_with_novel_spam.csv"
df_test_new = pd.read_csv(test_path)

X_test_raw = df_test_new['final_content']
y_test = df_test_new['label']

print(f"新测试集大小: {len(df_test_new)}")
print(f"新测试集类别分布:\n{y_test.value_counts()}")

# ==================== 3. TF-IDF 向量化 ====================
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

print(f"TF-IDF 特征维度: {X_train.shape[1]}")

# ==================== 4. 训练 SVM（充足样本基线） ====================
svm = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,
    random_state=42
)
svm.fit(X_train, y_train)

# ==================== 5. 在新测试集上评估 ====================
y_pred = svm.predict(X_test)

print("\n==================== 充足样本 SVM 分类报告（含新型诈骗） ====================")
print(classification_report(y_test, y_pred, target_names=['正常短信', '垃圾短信']))

print("混淆矩阵:")
print(confusion_matrix(y_test, y_pred))