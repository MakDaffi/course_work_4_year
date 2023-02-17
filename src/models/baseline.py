import pandas as pd
import utils

def popular(train, n):
    recomnd = []
    for i, _ in train.groupby('article_id').size().nlargest(n).items():
        recomnd.append(i)
    return recomnd

