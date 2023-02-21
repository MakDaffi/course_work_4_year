import pandas as pd
import utils.utils as utls
from baseline import recommend_popular
from implicit.nearest_neighbours import TFIDFRecommender

class TFIDFRec(utls.model):
    def fit(self):
        self.model.fit(self.user_items, True)

    
    def recommend(self, user, n=12):
        usr = self.user_id_to_uid[user]
        recommendations = self.model.recommend(usr, self.user_items[usr], N=n)[0]
        return [self.iid_to_item_id[i] for i in recommendations]

if __name__ == "__main__" :
    print("---------------------Loading data---------------------")
    users = pd.read_csv('../../data/processed_data/customers.csv')
    items = pd.read_csv('../../data/processed_data/articles.csv')
    transactions = pd.read_csv('../../data/processed_data/transactions.csv')
    print("-------------------Train/test split-------------------")
    train, validation = utls.train_test_split(transactions, items, users, '2020-07-01')
    print("------------Get true values on validation-------------")
    true_dct = utls.get_y_true(users.customer_id.unique(), validation)
    print("-----------------------Succsses-----------------------")
    print("--------------------Training model--------------------")
    recommender = TFIDFRec(train, TFIDFRecommender())
    recommender.fit()
    print("-----------------------Succsses-----------------------")
    print("-------------------Evaluation model-------------------")
    pred_dct = {}
    mapk = 0
    popular = recommend_popular(train, 12)
    for i in users.customer_id.unique():
        if i in recommender.user_id_to_uid:
            pred_dct[i] = recommender.recommend(i)
        else:
            pred_dct[i] = popular         
        mapk += utls.apk(true_dct[i], pred_dct[i], 12)
    print("MAP12 on validation:", mapk / len(users.customer_id.unique()))
    print("-----------------------Succsses-----------------------")
    print("------------------Prepair submission------------------")
    sub = pd.read_csv('../../data/sample_submission.csv')
    popular = ' '.join(["0" + str(elem) for elem in popular])
    sub.prediction = sub.customer_id.apply(lambda x: ' '.join(["0" + str(elem) for elem in pred_dct[x]]) if x in pred_dct else popular)
    sub.to_csv('../../data/submissions/tfidf_submission.csv', index=False)

