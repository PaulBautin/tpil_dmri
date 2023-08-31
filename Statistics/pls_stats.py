
from __future__ import division

import pandas as pd
import os
import numpy as np
import scipy.stats as stats
from sklearn import cross_decomposition, utils
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils.extmath import randomized_svd
from sklearn.utils.validation import check_X_y, check_random_state

def svd(crosscov, n_components=None, seed=None):
    """
    Calculates the SVD of `crosscov` and returns singular vectors/values

    Parameters
    ----------
    crosscov : (B, T) array_like
        Cross-covariance (or cross-correlation) matrix to be decomposed
    n_components : int, optional
        Number of components to retain from decomposition
    seed : {int, :obj:`numpy.random.RandomState`, None}, optional
        Seed for random number generation. Default: None

    Returns
    -------
    U : (B, L) `numpy.ndarray`
        Left singular vectors from singular value decomposition
    d : (L, L) `numpy.ndarray`
        Diagonal array of singular values from singular value decomposition
    V : (J, L) `numpy.ndarray`
        Right singular vectors from singular value decomposition
    """

    seed = check_random_state(seed)
    crosscov = np.asanyarray(crosscov)

    if n_components is None:
        n_components = min(crosscov.shape)
    elif not isinstance(n_components, int):
        raise TypeError('Provided `n_components` {} must be of type int'
                        .format(n_components))

    # run most computationally efficient SVD
    if crosscov.shape[0] <= crosscov.shape[1]:
        U, d, V = randomized_svd(crosscov.T, n_components=n_components,
                                 random_state=seed, transpose=False)
        V = V.T
    else:
        V, d, U = randomized_svd(crosscov, n_components=n_components,
                                 random_state=seed, transpose=False)
        U = U.T

    return U, np.diag(d), V


def normalize(X, axis=0):
    """
    Normalizes `X` along `axis`

    Utilizes Frobenius norm (or Hilbert-Schmidt norm / `L_{p,q}` norm where
    `p=q=2`)

    Parameters
    ----------
    X : (S, B) array_like
        Input array
    axis : int, optional
        Axis for normalization. Default: 0

    Returns
    -------
    normed : (S, B) `numpy.ndarray`
        Normalized `X`
    """

    normed = np.array(X)
    normal_base = np.linalg.norm(normed, axis=axis, keepdims=True)
    # avoid DivideByZero errors
    zero_items = np.where(normal_base == 0)
    normal_base[zero_items] = 1
    # normalize and re-set zero_items to 0
    normed = normed / normal_base
    normed[zero_items] = 0

    return normed


def xcorr(X, Y, norm=False, covariance=False):
    """
    Calculates the cross-covariance matrix of `X` and `Y`

    Parameters
    ----------
    X : (S, B) array_like
        Input matrix, where `S` is samples and `B` is features.
    Y : (S, T) array_like, optional
        Input matrix, where `S` is samples and `T` is features.
    norm : bool, optional
        Whether to normalize `X` and `Y` (i.e., sum of squares = 1). Default:
        False
    covariance : bool, optional
        Whether to calculate the cross-covariance matrix instead of the cross-
        correlation matrix. Default: False

    Returns
    -------
    xprod : (T, B) `numpy.ndarray`
        Cross-covariance of `X` and `Y`
    """

    check_X_y(X, Y, multi_output=True)

    # we could just use scipy.stats zscore but if we do this we retain the
    # original data structure; if pandas dataframes were given, a dataframe
    # will be returned
    if not covariance:
        Xn = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        Yn = (Y - Y.mean(axis=0)) / Y.std(axis=0, ddof=1)
    else:
        Xn, Yn = X - X.mean(0, keepdims=True), Y - Y.mean(0, keepdims=True)

    if norm:
        Xn, Yn = normalize(Xn), normalize(Yn)

    xprod = (Yn.T @ Xn) / (len(Xn) - 1)

    return xprod


def varexp(singular):
    """
    Calculates the variance explained by values in `singular`

    Parameters
    ----------
    singular : (L, L) array_like
        Singular values from singular value decomposition

    Returns
    -------
    varexp : (L, L) `numpy.ndarray`
        Variance explained
    """

    if singular.ndim != 2:
        raise ValueError('Provided `singular` array must be a square diagonal '
                         'matrix, not array of shape {}'
                         .format(singular.shape))

    return np.diag(np.diag(singular)**2 / np.sum(np.diag(singular)**2))


def get_behavioral_stats(df_participants):
    """
    get behavioral stats
    """
    ## behavior metrics
    df_clbp = pd.read_csv("/home/pabaua/dev_tpil/data/Données_Paul_v2_clbp.csv", sep=',')
    df_con = pd.read_csv("/home/pabaua/dev_tpil/data/Données_Paul_v2_control.csv", sep=',')
    df = pd.concat([df_clbp, df_con])
    df = df.drop(df.filter(regex='PD_score', axis=1).columns, axis=1)
    df = df.drop(df.filter(regex='Score_SC', axis=1).columns, axis=1)

    # average scores over the 3 sessions
    df['STAI-S'] = df.filter(like='Score_STAI-S').mean(axis=1)
    df['STAI-T'] = df.filter(like='Score_STAI-T').mean(axis=1)
    df['BECK'] = df.filter(like='Score BECK').mean(axis=1)
    df['PCS'] = df.filter(like='Score PCS').mean(axis=1)

    df = df.drop(df.filter(regex='BL', axis=1).columns, axis=1)
    df = df.drop(df.filter(regex='2M', axis=1).columns, axis=1)
    df = df.drop(df.filter(regex='4M', axis=1).columns, axis=1)

    df = df.drop('Groupe', axis=1)
    df['ID'] = 'sub-pl' + df['ID'].str.split('PT_PL').str[-1]
    df = df.rename(columns={'ID':'subject'})
    df = pd.merge(df_participants, df).sort_values(['group','subject'])
    return df


def get_demographic_measures():
    ## demographic metrics
    path_demo = os.path.abspath(os.path.expanduser("/home/pabaua/dev_tpil/data/22-08-19_dMRI_CLBP_BIDS/participants.tsv"))
    df_participants = pd.read_csv(path_demo, sep='\t')
    df_participants = df_participants.rename(columns={'participant_id':'subject'})
    df_participants['subject'] = 'sub-pl' + df_participants['subject'].str.split('sub-').str[-1]
    return df_participants

def main():
    """
    main function, gather stats and call plots
    """
    df_participants = get_demographic_measures()
    df_behavior = get_behavioral_stats(df_participants)
    out_dir = os.path.abspath("/home/pabaua/example_data/example_data/")

    fname_con = 'df_control_schaefer_volume_sub_norm_neuro.pkl'
    fname_clbp = 'df_clbp_schaefer_volume_sub_norm_neuro.pkl'
    out_dir = os.path.abspath("/home/pabaua/example_data/example_data/")
    df_energy = pd.concat([pd.read_pickle(os.path.join(out_dir, fname_con)),
                           pd.read_pickle(os.path.join(out_dir, fname_clbp))])
    df_energy = df_energy.set_index('session').loc['v1'].reset_index()
    df_energy = df_energy.set_index(['session', 'subject','x0']).unstack('x0')
    df_energy.columns = df_energy.columns.map('|'.join).str.strip('|')
    df_energy = df_energy.reset_index().drop('session', axis=1).set_index('subject')
    df_energy = df_energy.apply(lambda x: stats.zscore(x), axis=0)
    df_behavior = df_behavior.drop(['age', 'sex', 'group'], axis=1).set_index('subject').apply(lambda x: stats.zscore(x), axis=0)
    df_behavior = df_behavior.drop('sub-pl004', axis=0)
    #df_participants = df_participants.set_index('subject')
    #df_metric = df_participants.join(df_energy).sort_values(['group','subject'])

    df_energy = df_energy.sort_values('subject')
    df_behavior = df_behavior.sort_values('subject')
    print(df_energy.index.equals(df_behavior.index))
    pls = cross_decomposition.PLSCanonical(n_components=4, algorithm='svd')
    x_scores, y_scores = pls.fit(df_energy, df_behavior).transform(df_energy, df_behavior)

    # print(x_scores.shape)
    # print(y_scores.shape)
    # score = pls.score(df_energy, df_behavior)
    # print(score)
    actual_Rs = np.array([stats.pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in zip(x_scores.T, y_scores.T)])

    # permute the behavioral data

    permuted_Rs = [pls.fit_transform(utils.shuffle(df_energy, random_state=n), df_behavior) for n in range(500)]
    permuted_Rs = np.array([np.array([stats.pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in zip(x_scores.T, y_scores.T)]) for x_scores, y_scores in permuted_Rs])
    print(permuted_Rs.shape)

    # # plot covariance explained
    # plt.figure(figsize = (6,5))
    # #sns.set(style="ticks",font_scale=2)
    # p1 = sns.boxplot(permuted_Rs * 100, color='white')
    # p2 = sns.scatterplot(x=range(4), y=actual_Rs*100, s=80, label='effect size')
    # plt.ylabel("% covariance accounted for")
    # plt.xlabel("latent variable")
    # p1.xaxis.tick_top()
    # plt.legend(frameon=False)
    # plt.tight_layout()
    # plt.show()


    crosscov = xcorr(X=df_energy, Y=df_behavior)
    U, d, V = svd(crosscov, n_components=4, seed=None)
    variance_exp = np.diag(varexp(d))

    from scipy.stats import pearsonr
    XU = np.dot(df_energy, U)
    YV = np.dot(df_behavior, V)
    r, p = pearsonr(XU[:, 0], YV[:, 0])
    print('r = {:.4f}, p = {:.4f}'.format(r, p))

    permuted_crosscov = [xcorr(utils.shuffle(df_energy, random_state=n).values, df_behavior.values) for n in range(500)]
    permuted_svd = [svd(permuted_crosscov[n], n_components=4, seed=None) for n in range(500)]
    permuted_varexp = np.array([np.diag(varexp(permuted_svd[n][1])) for n in range(500)])
    print(permuted_varexp)

    # plot covariance explained
    plt.figure(figsize = (6,5))
    #sns.set(style="ticks",font_scale=2)
    p1 = sns.boxplot(permuted_varexp * 100, color='white')
    p2 = sns.scatterplot(x=range(4), y=variance_exp*100, s=80, label='effect size')
    plt.ylabel("% covariance accounted for")
    plt.xlabel("latent variable")
    p1.xaxis.tick_top()
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()
    

 

if __name__ == "__main__":
    main()