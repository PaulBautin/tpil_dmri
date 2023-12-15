
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


from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer



def main():
    """
    main function, gather stats and call plots
    """
    path_demographics = os.path.abspath(os.path.expanduser("/home/pabaua/dev_tpil/data/22-08-19_dMRI_CLBP_BIDS/participants.tsv"))
    df_participants = pd.read_csv(path_demographics, sep='\t')
    df_participants = df_participants.rename(columns={'participant_id':'subject'})
    df_participants['Subject'] = 'sub-pl' + df_participants['subject'].str.split('sub-').str[-1]
    df_participants = df_participants.rename(columns={'age': 'Age_in_Yrs', 'sex': 'Gender'})[['Subject', 'Age_in_Yrs', 'Gender']].set_index('Subject')
    df_participants = df_participants[df_participants['Age_in_Yrs'] <= 36]

    path_demographics_hcp = os.path.abspath(os.path.expanduser("/home/pabaua/Downloads/RESTRICTED_pascaltetreault_12_6_2023_13_49_14.csv"))
    df_hcp_res = pd.read_csv(path_demographics_hcp, sep=',')
    path_demographics_hcp = os.path.abspath(os.path.expanduser("/home/pabaua/Downloads/unrestricted_baup2501_12_6_2023_12_48_21.csv"))
    df_hcp = pd.read_csv(path_demographics_hcp, sep=',')
    df_hcp = pd.merge(df_hcp_res, df_hcp, on='Subject')

    path_subjects = os.path.abspath(os.path.expanduser("/home/pabaua/Downloads/participants.txt"))
    df_subjects = pd.read_csv(path_subjects, sep=' ', header=None)
    df_subjects.columns = ['subject_scil']

    df_hcp = df_hcp[df_hcp['Subject'].isin(df_subjects['subject_scil'])]
    df_hcp = df_hcp[['Subject', 'Age_in_Yrs', 'Gender']].set_index('Subject')



    # Encode the Gender column
    numerical_columns_selector = selector(dtype_exclude=object)
    categorical_columns_selector = selector(dtype_include=object)
    numerical_columns = numerical_columns_selector(df_hcp)
    categorical_columns = categorical_columns_selector(df_hcp)
    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numerical_preprocessor = StandardScaler()

    preprocessor = ColumnTransformer(
        [("categorical", categorical_preprocessor, categorical_columns),
        ("numerical", numerical_preprocessor, numerical_columns)],
        remainder="passthrough",
    ).set_output(transform="pandas")
    df_hcp = preprocessor.fit_transform(df_hcp)
    df_participants = preprocessor.fit_transform(df_participants)
    print(df_participants)

    # Using Nearest Neighbors
    neighbors = NearestNeighbors(n_neighbors=1)
    neighbors.fit(df_hcp)

    # Find the nearest neighbor for each row in df1 with remises
    #distances, indices = neighbors.kneighbors(df_participants)
    #df_participants['Closest Match'] = list(df_hcp.iloc[indices.flatten()].reset_index()['Subject'])


    # Match subjects uniquely without remises
    matches = []
    distances = []
    df_hcp_copy = df_hcp.copy()
    for index, row in df_participants.iterrows():
        neighbors.fit(df_hcp_copy[['numerical__Age_in_Yrs', 'categorical__Gender_M', 'categorical__Gender_F']])
        distance, index = neighbors.kneighbors([row[['numerical__Age_in_Yrs', 'categorical__Gender_M', 'categorical__Gender_F']]])
        closest = df_hcp_copy.iloc[index[0]]
        matches.append(closest.index.values[0])
        df_hcp_copy = df_hcp_copy.drop(closest.index)

    df_participants['Unique Match'] = matches
    df_participants = df_participants.reset_index()
    df_hcp = df_hcp.reset_index()
    df_participants = pd.merge(df_participants, df_hcp, how='inner', left_on=['Unique Match'], right_on=['Subject'], suffixes=['', '_hcp'])


    print(df_participants)
    sns.barplot(data=df_participants, y='Subject', x='numerical__Age_in_Yrs_hcp')
    # Labels and Legend
    plt.xlabel('Age')
    plt.ylabel('Gender')
    plt.title('Nearest Neighbors Visualization')
    plt.legend()
    plt.show()
 

if __name__ == "__main__":
    main()