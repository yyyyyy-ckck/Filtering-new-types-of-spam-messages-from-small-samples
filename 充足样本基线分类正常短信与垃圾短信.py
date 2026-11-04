import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ==================== 1. 加载全量数据 ====================
file_path = r"C:\Users\yj178\Desktop\ei论文\final_cleaned_data.csv"
df = pd.read_csv(file_path)

# 文本列和标签列
X_raw = df['final_content']
y = df['label']

print(f"数据集总大小: {len(df)}")
print(f"类别分布:\n{y.value_counts()}")

# ==================== 2. 划分训练集与测试集 (7:3) ====================
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y,
    test_size=0.3,
    stratify=y,        # 分层抽样，保持训练/测试集中类别比例一致
    random_state=42
)

print(f"训练集大小: {len(X_train_raw)}")
print(f"测试集大小: {len(X_test_raw)}")

# ==================== 3. TF-IDF 向量化 ====================
vectorizer = TfidfVectorizer(
    max_features=5000,        # 最多保留5000个最重要的词
    ngram_range=(1, 2),       # 使用单个词和两个词的组合
    min_df=2,                 # 至少在2条短信中出现才保留，避免罕见词
    sublinear_tf=True         # 使用 1+log(tf) 缩放缓和高频词影响
)

# 只用训练集拟合词表，然后转换训练集和测试集
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

print(f"TF-IDF 特征维度: {X_train.shape[1]}")

# ==================== 4. 训练 SVM（充足样本基线） ====================
svm = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',            # 自动设置 gamma
    probability=True,
    random_state=42
)
svm.fit(X_train, y_train)

# ==================== 5. 测试集评估 ====================
y_pred = svm.predict(X_test)

print("\n==================== 充足样本 SVM 分类报告 ====================")
print(classification_report(y_test, y_pred, target_names=['正常短信', '垃圾短信']))

print("混淆矩阵:")
print(confusion_matrix(y_test, y_pred))