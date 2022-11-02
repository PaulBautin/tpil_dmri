# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn_pandas import DataFrameMapper


def fit_pca(df, index):
    """
    Compute PCA metrics
    :param df_con: Control group dataframe
    :param df_clbp: Chronic Lower Back Pain group dataframe
    :param labels: labels to index dataframe
    :param diff: Compute PCA metrics on raw data diff=None, on difference of metrics pert tract: diff="per_tract"
    or on difference of metrics per point: diff="per_point"
    :param output_metrics: Number and name of PCA metrics to output
    :return df_pca: dataframe containing pca metrics
    :return pca: output of sklearn.decomposition.PCA
    :return x_norm: dataframe of the normalized initial metrics
    """
    # Isolate and standardize metrics (remove mean and scale to unit variance)
    df = df.set_index(index).dropna()
    x_norm = StandardScaler().fit_transform(df.values)
    df_x_norm = pd.DataFrame(x_norm, index=df.index, columns=df.columns)
    pca = PCA(n_components=2)
    pca.fit(x_norm)
    return pca, df_x_norm


def apply_pca(pca, df, output_metrics=['PCA_1', 'PCA_2']):
    """
    Compute PCA metrics
    :param df_con: Control group dataframe
    :param df_clbp: Chronic Lower Back Pain group dataframe
    :param labels: labels to index dataframe
    :param diff: Compute PCA metrics on raw data diff=None, on difference of metrics pert tract: diff="per_tract"
    or on difference of metrics per point: diff="per_point"
    :param output_metrics: Number and name of PCA metrics to output
    :return df_pca: dataframe containing pca metrics
    :return pca: output of sklearn.decomposition.PCA
    :return x_norm: dataframe of the normalized initial metrics
    """
    df_pca = pd.DataFrame(data=pca.transform(df.values), index=df.index, columns=output_metrics)
    return df_pca.reset_index()
