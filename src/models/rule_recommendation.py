import pandas as pd
import utils.utils as utls
from baseline import recommend_popular

def recommend_popular_with_rules(train, n=12):
    recommend = {
        0.0: [[i for i, _ in train.loc[(train['sex_y'] == 0) & ((train['target_age_groupe'] == 0) | (train['target_age_groupe'] == 1))].groupby('article_id').size().nlargest(n).items()], [i for i, _ in train.loc[(train['sex_y'] == 0) & ((train['target_age_groupe'] == 2) | (train['target_age_groupe'] == 3))].groupby('article_id').size().nlargest(n).items()]], 
        1.0: [[i for i, _ in train.loc[(train['sex_y'] == 1) & ((train['target_age_groupe'] == 0) | (train['target_age_groupe'] == 1))].groupby('article_id').size().nlargest(n).items()], [i for i, _ in train.loc[(train['sex_y'] == 1) & ((train['target_age_groupe'] == 2) | (train['target_age_groupe'] == 3))].groupby('article_id').size().nlargest(n).items()]], 
        0.5: [[i for i, _ in train.loc[(train['sex_y'] == 3) & ((train['target_age_groupe'] == 0) | (train['target_age_groupe'] == 1))].groupby('article_id').size().nlargest(n).items()], [i for i, _ in train.loc[(train['sex_y'] == 3) & ((train['target_age_groupe'] == 2) | (train['target_age_groupe'] == 3))].groupby('article_id').size().nlargest(n).items()]]
    }
    return recommend

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
    true_dct = utls.get_y_true(users.customer_id.unique(), validation)
    recommend = recommend_popular_with_rules(train, 12)
    print("-----------------------Succsses-----------------------")
    print("-------------------Evaluation model-------------------")
    mapk = 0
    pred_dct = {}
    mapk = 0
    popular = recommend_popular(train, 12)
    age_group = {0:0,1:0,2:1,3:1}
    for i,j in zip(users.customer_id.unique(), users.apply(lambda x:recommend[x.sex][age_group[x.age_group]], axis=1)):
        pred_dct[i] = j
        mapk += utls.apk(true_dct[i], pred_dct[i], 12)
    print("MAP12 on validation:", mapk / len(users.customer_id.unique()))
    print("-----------------------Succsses-----------------------")
    print("------------------Prepair submission------------------")
    sub = pd.read_csv('../../data/sample_submission.csv')
    popular = ' '.join(["0" + str(elem) for elem in popular])
    sub.prediction = sub.customer_id.apply(lambda x: ' '.join(["0" + str(elem) for elem in pred_dct[x]]) if x in pred_dct else popular)
    sub.to_csv('../../data/submissions/rule_v2_submission.csv', index=False)
