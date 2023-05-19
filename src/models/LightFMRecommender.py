import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k

def feature_colon_value(features, values):
    result = []
    ll = features
    aa = values
    for x,y in zip(ll,aa):
        res = str(x) +":"+ str(y)
        result.append(res)
    return result

class LightFMrecommender:
    def __init__(self, users, items, transactions, model, use_features=False):
        uf = []
        col = ['Active']*len(users.Active.unique()) + ['club_member_status']*len(users.club_member_status.unique()) + ['fashion_news_frequency']*len(users.fashion_news_frequency.unique()) + ['age_group']*len(users.age_group.unique()) + ['sex']*len(users.sex.unique())  + ['baby']*len(users.baby.unique())
        unique_f1 = list(users.Active.unique()) + list(users.club_member_status.unique()) + list(users.fashion_news_frequency.unique()) + list(users.age_group.unique()) + list(users.sex.unique()) + list(users.baby.unique())
        for x,y in zip(col, unique_f1):
            res = str(x)+ ":" +str(y)
            uf.append(res)
        fi = []
        self.dataset = Dataset()
        self.dataset.fit(users=users['customer_id'], 
                    items=items['article_id'],
                    user_features = uf,
                    item_features=fi)

        num_users, num_topics = self.dataset.interactions_shape()
        print(f'Number of users: {num_users}, Number of topics: {num_topics}.')
        item_features = ["product_code","product_type_no","graphical_appearance_no","colour_group_code","section_no","garment_group_no","season","sex","target_age_groupe"]
        col = []
        unique_f1 = []
        for i in item_features:
            col += [i] * len(items[i].unique())
            unique_f1 += list(items[i].unique())
        for x,y in zip(col, unique_f1):
            res = str(x)+ ":" +str(y)
            fi.append(res)
        features = ['Active', 'club_member_status', 'fashion_news_frequency', 'age_group', 'sex', 'baby']
        ad_subset = users[features]
        ad_list = [list(x) for x in ad_subset.values]
        feature_list = []
        for item in ad_list:
            feature_list.append(feature_colon_value(features, item))
        ad_subset = items[item_features]
        ad_list = [list(x) for x in ad_subset.values]
        feature_list1 = []
        for item in ad_list:
            feature_list1.append(feature_colon_value(item_features, item))
        user_tuple = list(zip(users.customer_id, feature_list))
        self.user_features = self.dataset.build_user_features(user_tuple, normalize= False)
        item_tuple = list(zip(items.article_id, feature_list1))
        self.item_features = self.dataset.build_item_features(item_tuple, normalize= False)
        train_set = transactions[transactions.t_dat<='2020-9-15']
        val_set = transactions[(transactions.t_dat>='2020-9-16')&(transactions.t_dat<='2020-9-22')]

        self.interactions, _ = self.dataset.build_interactions(train_set.iloc[:, 1:3].values)
        self.val_interactions, _ = self.dataset.build_interactions(val_set.iloc[:, 1:3].values)
        self.model = model
        self.use_features = use_features

    def fit(self, num_epochs, verbose, num_threads):
        if self.use_features:
            self.model.fit(interactions=self.interactions, epochs=num_epochs, verbose=verbose, user_features=self.use_features, item_features=self.item_features, num_threads=num_threads)
        else:
            self.model.fit(interactions=self.interactions, epochs=num_epochs, verbose=verbose, num_threads=num_threads)

    def validation(self, k):
        if self.use_features:
            print(precision_at_k(self.model, self.val_interactions, user_features=self.use_features, item_features=self.item_features, k=k).mean())
        else:
            print(precision_at_k(self.model, self.val_interactions, k=k).mean())

    def predict(self, usr):
        _, _, iid_map, _ = self.dataset.mapping()
        inv_iid_map = {v:k for k, v in iid_map.items()}
        m_opt = self.model.predict(np.array([usr] * len(iid_map)), np.array(list(iid_map.values())))
        pred = np.argsort(-m_opt)[:12]
        return ' '.join([inv_iid_map[p] for p in pred]).strip()