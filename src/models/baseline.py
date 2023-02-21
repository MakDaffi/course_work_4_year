import pandas as pd
import utils.utils as utls

def recommend_popular(train, n):
    recomnd = []
    for i, _ in train.groupby('article_id').size().nlargest(n).items():
        recomnd.append(i)
    return recomnd

if __name__ == "__main__" :
    print("---------------------Loading data---------------------")
    users = pd.read_csv('../../data/processed_data/customers.csv')
    items = pd.read_csv('../../data/processed_data/articles.csv')
    transactions = pd.read_csv('../../data/processed_data/transactions.csv')
    print("-------------------Train/test split-------------------")
    train, validation = utls.train_test_split(transactions, items, users, '2020-06-01')
    print("------------Get true values on validation-------------")
    dct = utls.get_y_true(users.customer_id.unique(), validation)
    print("-----------------------Succsses-----------------------")
    print("--------------------Training model--------------------")
    dct = utls.get_y_true(users.customer_id.unique(), validation)
    recommend = recommend_popular(train, 12)
    print("-----------------------Succsses-----------------------")
    print("-------------------Evaluation model-------------------")
    mapk = 0
    for k, v in dct.items():
        mapk += utls.apk(v, recommend, 12)
    print("MAP12 on validation:", mapk / len(users.customer_id.unique()))
    print("-----------------------Succsses-----------------------")
    sub = pd.read_csv('../../data/sample_submission.csv')
    ans = '0'
    for i in recommend:
        ans = ans + str(i) + ' '
    ans = ans[:len(ans) - 1]
    sub.prediction = ans
    sub.to_csv('../../data/submissions/baseline_submission.csv', index=False)