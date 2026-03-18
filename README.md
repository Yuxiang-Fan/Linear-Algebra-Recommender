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
