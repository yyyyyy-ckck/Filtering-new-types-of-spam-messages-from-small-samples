# Filtering-new-types-of-spam-messages-from-small-samples
New types of spam messages continue to emerge while sample data remains scarce. This paper proposes a two-stage cascaded few-shot learning framework to filter out newly emerging spam messages with limited samples.
# Cascaded Few-Shot Learning for Emerging SMS Spam Detection

# 基于级联架构与 AI 特征工程的小样本垃圾短信检测


## 1. 项目背景 (Motivation)

针对新型垃圾短信爆发初期标注数据极度匮乏（仅20条）的问题，提出两级级联小样本过滤框架。



## 2. 核心架构 (Methodology)

- **模型一（通用拦截）**：TF-IDF + RBF-SVM，处理常规历史垃圾短信。

- **模型二（小样本补漏）**：BERT 语义特征提取 + SMOTE 过采样 + SVM，专门识别模型一漏判的新型诈骗。

- **级联决策**：模型一判为垃圾直接拦截，判为正常则交由模型二二次筛查。



## 3. 核心创新 (Innovations)

1. **级联架构**：不破坏原有拦截能力，以旁路补漏的方式快速响应新威胁。

2. **AI 特征工程**：对比 TF-IDF，BERT 语义向量有效弥合了语义鸿沟，在小样本下性能提升显著。

3. **SMOTE 增强**：解决 20 条正样本与 400 条负样本的极端不平衡问题。



## 4. 实验结果 (Results)

| 模型 | 正常短信精确率 | 垃圾短信召回率 |

| --- | --- | --- |

| 模型一（基线） | 0.83 | 0.94 |

| 级联 + TF-IDF | 0.83 | 0.94 |

| **级联 + BERT (本文)** | **0.91** | **0.97** |


*消融实验证明：移除 SMOTE 后漏判数翻倍；移除级联框架（纯BERT）后精确率暴跌至 0.32。*



## 5. 运行指南 (Quick Start)

1. `pip install -r requirements.txt`

2. 下载 `bert-base-chinese` 模型至 `./bert-base-chinese` 目录。

3. 运行 `python src/cascade\_bert.py`

