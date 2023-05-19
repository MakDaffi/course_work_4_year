import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix


class model:
    def __init__(self, transactions, model):
        transactions["uid"] = transactions["customer_id"].astype("category")
        transactions["uid"] = transactions["uid"].cat.codes

        transactions["iid"] = transactions["article_id"].astype("category")
        transactions["iid"] = transactions["iid"].cat.codes
        
        self.iid_to_item_id = transactions[["iid", "article_id"]].drop_duplicates()\
            .set_index("iid").to_dict()["article_id"]
        self.item_id_to_iid = transactions[["iid", "article_id"]].drop_duplicates()\
            .set_index("article_id").to_dict()["iid"]

        self.uid_to_user_id = transactions[["uid", "customer_id"]].drop_duplicates()\
            .set_index("uid").to_dict()["customer_id"]
        self.user_id_to_uid = transactions[["uid", "customer_id"]].drop_duplicates()\
            .set_index("customer_id").to_dict()["uid"]

        indptr = []
        indices = []
        data = []

        for i,j in transactions[['uid', 'iid']].groupby(['uid', 'iid']).size().items():
            indptr.append(i[0])
            indices.append(i[1])
            data.append(j)

        self.user_items = csr_matrix((np.array(data).astype(float),
                                      (np.array(indptr), np.array(indices))))
        self.item_users = csr_matrix((np.array(data).astype(float),
                                      (np.array(indices), np.array(indptr))))
        self.model = model

def train_test_split(transactions, items, users, date):
    all_data = transactions.merge(users, on='customer_id', 
                                  how='inner').merge(items, on='article_id', how='inner')
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
    for k, v in df[['customer_id', 'article_id']].groupby(['customer_id', 
                                                           'article_id']).size().items():
        for _ in range(v):
            dct[k[0]].append(k[1])
    return dct