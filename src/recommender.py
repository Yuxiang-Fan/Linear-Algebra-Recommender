from surprise import KNNBasic, accuracy
from surprise.model_selection import train_test_split
from collections import defaultdict

class MovieRecommender:
    def __init__(self, sim_name='cosine', user_based=True):
        # 采用基于内积空间的余弦相似度
        self.sim_options = {'name': sim_name, 'user_based': user_based}
        self.algo = KNNBasic(sim_options=self.sim_options)

    def train(self, data, test_size=0.2):
        """训练模型并评估 """
        trainset, testset = train_test_split(data, test_size=test_size)
        self.algo.fit(trainset)
        predictions = self.algo.test(testset)
        rmse = accuracy.rmse(predictions)
        return rmse, predictions

    def get_top_n(self, predictions, n=10):
        """获取前 N 项推荐 """
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]
        return top_n