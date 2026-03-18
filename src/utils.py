import pandas as pd
import os
from surprise import Dataset, Reader

def load_movie_data(data_dir):
    """加载并预处理电影和评分数据"""
    ratings_path = os.path.join(data_dir, "ratings.csv")
    movies_path = os.path.join(data_dir, "movies.csv")

    if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
        raise FileNotFoundError(f"数据文件未找到，请确保 ratings.csv 和 movies.csv 位于 {data_dir}")

    # 加载数据
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    
    # 定义评分范围
    reader = Reader(rating_scale=(0.5, 5.0))
    # 将数据加载到 surprise 格式 
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    
    return data, movies