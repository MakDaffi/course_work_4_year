import pandas as pd
from tqdm import tqdm

def train_test_split(transactions, items, users, date):
    all_data = transactions.merge(users, on='customer_id', how='inner').merge(items, on='article_id', how='inner')
    train = all_data.loc[all_data.t_dat <= date]
    test = all_data.loc[all_data.t_dat > date]
    return train, test

def pk(y_true, y_pred, k):
    count = 0
    for i in y_pred:
        if i in y_true:
            count+=1
    return count / k

def apk(y_true, y_pred, k):
    count = 0
    for i in range(len(y_pred)):
        count += int(y_pred[i] in y_true) * pk(y_true, y_pred[:i+1], i+1)
    return count / k

def get_y_true(users, df):
    dct = {i:[] for i in users}
    for k, v in df[['customer_id', 'article_id']].groupby(['customer_id', 'article_id']).size().items():
        for _ in range(v):
            dct[k[0]].append(k[1])
    return dct