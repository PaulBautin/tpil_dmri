# -*- coding: utf-8 -*-
import pandas as pd
from scipy.stats import ttest_ind


def t_test_longitudinal(df):
    # filtering dataframe
    df = df.set_index(['subject', 'tract', 'point'])
    dict = {'v1': df[df['session'] == "v1"],
            'v2': df[df['session'] == "v2"],
            'v3': df[df['session'] == "v3"]}
    df2 = pd.concat(dict, axis=1).reset_index().groupby('point')
    metrics = [i for i in df.keys() if "mean" in i or "PCA" in i]
    dict = {}
    for m in metrics:
        t_test = lambda x: ttest_ind(x['v1'][m], x['v2'][m], equal_var=True, nan_policy='omit')[1]
        dict[m] = df2.apply(t_test)
    return pd.concat(dict, axis=1)





