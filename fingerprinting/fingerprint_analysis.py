#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import seaborn as sns
import glob
import pandas as pd
from scipy import stats, signal
import numpy as np
import matplotlib.pyplot as plt



def find_files_with_common_name_structural(directory, common_name, id='conn'):
    file_paths = glob.glob(os.path.join(directory, common_name))
    dict_paths = {os.path.basename(os.path.dirname(os.path.dirname(fp))): fp for fp in file_paths}
    df = pd.DataFrame.from_dict(dict_paths, orient='index').reset_index().rename(columns={'index': 'participant_id', 0: 'path_'+id})
    df[['subject', 'session']] = df['participant_id'].str.rsplit('_ses-', n=1, expand=True)
    df[['subject', 'num']] = df['subject'].str.rsplit('pl0', n=1, expand=True)
    df['subject'] = df['subject'] + df['num']
    df['img_'+id] = df['path_'+id].apply(lambda x: pd.read_csv(x, header=None).values)
    df = df.drop(["participant_id", 'num', 'path_'+id], axis=1)
    return df


def main():
    """
    Main function to execute the pipeline for connectivity mapping and visualization.
    """
    # Load images into dataframe
    # directories
    directory_conn = '/home/pabaua/dev_tpil/results/results_connectflow/23-08-21_connectflow/all_schaefer/'

    # id='sc'
    # common_name = '*/Compute_Connectivity/sc_vol_normalized.csv'
    # df_img = find_files_with_common_name_structural(directory_conn, common_name, id=id)

    id='commit2'
    common_name = '*/Compute_Connectivity/commit2_weights.csv'
    df_commit_img = find_files_with_common_name_structural(directory_conn, common_name, id=id)
    df_img = df_commit_img
    #df_img = pd.merge(df_img, df_commit_img, on=['subject','session'])

    # id='len'
    # common_name = '*/Compute_Connectivity/len.csv'
    # df_len_img = find_files_with_common_name_structural(directory_conn, common_name, id=id)
    # df_img = pd.merge(df_img, df_len_img, on=['subject','session'])
    print(df_img)

    # Filter the DataFrame for v1 and v2 sessions
    v1_df_img = df_img[df_img['session'] == 'v1'].set_index('subject').drop(columns='session')
    v2_df_img = df_img[df_img['session'] == 'v2'].set_index('subject').drop(columns='session')
    common_subjects = v1_df_img.index.intersection(v2_df_img.index)

    # Initialize an empty DataFrame for the correlation matrix
    correlation_matrix = pd.DataFrame(np.nan, index=common_subjects, columns=common_subjects)

    # Compute correlations
    for v1_subject in common_subjects:
        for v2_subject in common_subjects:
            correlation_matrix.at[v1_subject, v2_subject] = stats.pearsonr(v1_df_img.loc[v1_subject].mean().flatten(), 
                                                                           v2_df_img.loc[v2_subject].mean().flatten())[0]

    # Display the correlation matrix
    df_dem = pd.read_excel('/home/pabaua/dev_tpil/data/Données_Paul_v2.xlsx', sheet_name=1).drop(columns='id')
    correlation_matrix = pd.merge(correlation_matrix, df_dem, on='subject').set_index('subject')
    correlation_matrix.index.name = 'v1_subject'
    correlation_matrix.columns.name = 'v2_subject'
    correlation_matrix = correlation_matrix.sort_values(by = ['type', 'v1_subject'], ascending = [False, True])

    print(correlation_matrix)
    correlation_matrix = correlation_matrix[correlation_matrix.index]
    sns.heatmap(correlation_matrix)
    plt.show()
    




if __name__ == "__main__":
    main()
