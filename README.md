# Linear-Algebra-Based Collaborative Filtering Engine

A Python-based movie recommendation system that implements user-based collaborative filtering. This project bridges the gap between abstract mathematical theory and practical machine learning engineering by applying **Inner Product Spaces** to compute similarities between users in high-dimensional sparse datasets.

## 🧠 Mathematical Theory: Inner Product Space

At the core of collaborative filtering lies the ability to measure how similar two users (or items) are. This project models users as vectors in a high-dimensional real vector space $V$. 

By defining an inner product $\langle u, v \rangle$, we can induce both the norm (length) of a user vector $\|u\| = \sqrt{\langle u, u \rangle}$ and the angle $\theta$ between two user vectors. The similarity is computed using the Cosine Similarity formula derived from the inner product space:

$$\cos\theta = \frac{\langle u, v \rangle}{\|u\| \|v\|}$$

Using this geometric interpretation, the `scikit-surprise` library calculates the distance between user rating vectors to find the K-Nearest Neighbors (KNN), which are then used to predict unseen movie ratings.

## 📁 Project Structure

The project is structured for readability and maintainability:

```text
Movie-Recommendation-System/
├── data/               # MovieLens dataset (ratings.csv, movies.csv, tags.csv)
├── src/                # Core source code
│   ├── __init__.py
│   ├── utils.py        # Data loading and preprocessing
│   └── recommender.py  # KNNBasic model and prediction logic
├── docs/               # Academic reports and mathematical proofs
├── main.py             # Application entry point
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
