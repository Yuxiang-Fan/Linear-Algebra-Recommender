# Collaborative Filtering Implementation based on Linear Algebra

This project provides a Python-based implementation of a user-based collaborative filtering recommendation system. It explores the application of inner product spaces to determine similarities between users within high-dimensional sparse datasets.

## 🧠 Mathematical Framework: Inner Product Space

The methodology of collaborative filtering relies on quantifying the similarity between distinct users or items. In this implementation, users are represented as vectors within a high-dimensional real vector space $V$. 

By defining an inner product $\langle u, v \rangle$, the model establishes both the norm (length) of a user vector $\|u\| = \sqrt{\langle u, u \rangle}$ and the angle $\theta$ between two user vectors. The similarity is derived from the Cosine Similarity formula within the inner product space:

$$\cos\theta = \frac{\langle u, v \rangle}{\|u\| \|v\|}$$

Based on this geometric interpretation, the `scikit-surprise` library is utilized to identify K-Nearest Neighbors (KNN) through the calculation of distances between user rating vectors, which subsequently facilitates the prediction of unseen movie ratings.

## 📁 Project Structure

The repository is organized to maintain a clear separation between data, logic, and documentation:

```text
Movie-Recommendation-System/
├── data/                # MovieLens dataset (ratings.csv, movies.csv, tags.csv)
├── src/                 # Core source code
│   ├── __init__.py
│   ├── utils.py         # Data loading and preprocessing logic
│   └── recommender.py   # KNNBasic model and prediction implementation
├── docs/                # Academic reports and mathematical derivations
├── main.py              # Application entry point
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

---

# 基于线性代数的协同过滤实现

本项目提供了一个基于 Python 的用户协同过滤推荐系统的实现。它探索了内积空间在高维稀疏数据集中确定用户间相似度的应用。

## 🧠 数学框架：内积空间

协同过滤的方法论依赖于量化不同用户或物品之间的相似性。在本实现中，用户被表示为高维实向量空间 $V$ 中的向量。

通过定义内积 $\langle u, v \rangle$，模型确立了用户向量的范数（长度） $\|u\| = \sqrt{\langle u, u \rangle}$ 以及两个用户向量之间的夹角 $\theta$。相似度由内积空间内的余弦相似度公式推导得出：

$$\cos\theta = \frac{\langle u, v \rangle}{\|u\| \|v\|}$$

基于这一几何解释，本项目利用 `scikit-surprise` 库，通过计算用户评分向量之间的距离来识别 K-近邻（KNN），从而实现对未观测电影评分的预测。

## 📁 项目结构

仓库的组织方式保持了数据、逻辑和文档之间的清晰分离：

```text
Movie-Recommendation-System/
├── data/                # MovieLens 数据集 (ratings.csv, movies.csv, tags.csv)
├── src/                 # 核心源代码
│   ├── __init__.py
│   ├── utils.py         # 数据加载与预处理逻辑
│   └── recommender.py   # KNNBasic 模型与预测实现
├── docs/                # 学术报告与数学推导
├── main.py              # 应用程序入口
├── requirements.txt     # 项目依赖
└── README.md            # 项目文档
```
