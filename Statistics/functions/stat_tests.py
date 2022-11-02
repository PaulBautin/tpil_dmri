# -*- coding: utf-8 -*-
import pandas as pd
from scipy.stats import ttest_ind


def t_test_longitudinal(df):
    # filtering dataframe
    df_l = df.set_index(['group_name', 'session', 'subject', 'tract', 'point']).unstack(level='session')
    metrics = df_l.keys().get_level_values(0).unique().tolist()
    df_l = df_l.groupby(['group_name', 'tract', 'point'])
    t_test = lambda x, m: ttest_ind(x[m]['v2'], x[m]['v3'], equal_var=True, nan_policy='omit')[1]
    t_values = {m: df_l.apply(t_test, m) for m in metrics}
    return pd.concat(t_values, axis=1)


def t_test_cs_per_session_per_point(df):
    # filtering dataframe
    df_cs = df.set_index(['group_name', 'session', 'subject', 'tract', 'point']).unstack(level='group_name')
    metrics = df_cs.keys().get_level_values(0).unique().tolist()
    df_cs = df_cs.groupby(['session', 'tract', 'point'])
    t_test = lambda x, m: ttest_ind(x[m]['con'], x[m]['clbp'], equal_var=True, nan_policy='omit')[1]
    t_values = {m: df_cs.apply(t_test, m) for m in metrics}
    return pd.concat(t_values, axis=1)

def t_test_cs_per_session(df):
    # filtering dataframe
    df_cs = df.set_index(['group_name', 'session', 'subject', 'tract', 'point']).unstack(level='group_name')
    metrics = df_cs.keys().get_level_values(0).unique().tolist()
    df_cs = df_cs.groupby(['session', 'tract'])
    t_test = lambda x, m: ttest_ind(x[m]['con'], x[m]['clbp'], equal_var=True, nan_policy='omit')[1]
    t_values = {m: df_cs.apply(t_test, m) for m in metrics}
    return pd.concat(t_values, axis=1)

def t_test_cs(df):
    # filtering dataframe
    df_cs = df.set_index(['group_name', 'session', 'subject', 'tract', 'point']).unstack(level='group_name')
    metrics = df_cs.keys().get_level_values(0).unique().tolist()
    df_cs = df_cs.groupby(['tract'])
    t_test = lambda x, m: ttest_ind(x[m]['con'], x[m]['clbp'], equal_var=True, nan_policy='omit')[1]
    t_values = {m: df_cs.apply(t_test, m) for m in metrics}
    return pd.concat(t_values, axis=1)

def t_test_cs_mean(df):
    # filtering dataframe
    df_cs = df.groupby(['group_name', 'session', 'subject', 'tract']).mean().unstack(level='group_name')
    metrics = df_cs.keys().get_level_values(0).unique().tolist()
    df_cs = df_cs.groupby(['tract'])
    print(df_cs.mean())
    t_test = lambda x, m: ttest_ind(x[m]['con'], x[m]['clbp'], equal_var=True, nan_policy='omit')[1]
    t_values = {m: df_cs.apply(t_test, m) for m in metrics}
    return pd.concat(t_values, axis=1)






