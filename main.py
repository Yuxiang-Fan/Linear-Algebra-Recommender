import os
from src.utils import load_movie_data
from src.recommender import MovieRecommender

# 定义基础路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

def main():
    # 1. 加载数据
    data, movies = load_movie_data(DATA_DIR)
    
    # 2. 训练模型
    recommender = MovieRecommender()
    rmse, predictions = recommender.train(data)
    print(f"模型评估完成，RMSE: {rmse}")
    
    # 3. 生成推荐
    top_n = recommender.get_top_n(predictions, n=10)
    # 此处可继续添加打印逻辑或输出到文件
    print("推荐生成成功！")

if __name__ == "__main__":
    main()